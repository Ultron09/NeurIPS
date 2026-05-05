import time
import torch
import torch.nn.functional as F
from tqdm import tqdm


def mixup_data(x, y, alpha=0.4):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_single_task(model, train_loader, val_loader, optimizer, t_idx, device='cuda',
                      ewc_module=None, agem_module=None, replay_buffer=None,
                      der_module=None, hat_module=None,
                      epochs=10, patience=3, mixup_alpha=0.4, label_smoothing=0.0):
    """
    Unified training logic with Cognitive Bifurcation:
    - ANTARA path: delegates entirely to model.train_step() (full cognitive loop)
    - Baseline path: raw SGD loop for EWC, REPLAY, A-GEM, DER++, HAT, NAIVE
    """
    total_step_time = 0.0
    total_steps = 0

    if hasattr(model, 'config') and hasattr(model.config, 'classes_per_task'):
        classes_per_task = model.config.classes_per_task
    elif hasattr(train_loader.dataset, 'classes') and not isinstance(train_loader.dataset, torch.utils.data.Subset):
        classes_per_task = len(train_loader.dataset.classes)
    else:
        classes_per_task = 10
    
    seen_classes = (t_idx + 1) * classes_per_task
    is_antara = hasattr(model, 'train_step') and hasattr(model, 'consolidate_memory')

    try:
        for epoch in range(epochs):
            model.train()
            print(f"Task {t_idx} Epoch {epoch} started...")
        for x, y in train_loader:
            start_time = time.time()
            x, y = x.to(device), y.to(device)

            if is_antara:
                if replay_buffer and len(replay_buffer) > 0:
                    n_replay = min(x.size(0), len(replay_buffer))
                    rx, ry = replay_buffer.sample(n_replay)
                    rx, ry = rx.to(device).float(), ry.to(device)
                    mixed_x = torch.cat([x.float(), rx])
                    mixed_y = torch.cat([y, ry])
                    result = model.train_step(mixed_x, target_data=mixed_y,
                                              task_id=t_idx,
                                              enable_dream=True,
                                              meta_step=True,
                                              record_stats=True)
                else:
                    result = model.train_step(x.float(), target_data=y,
                                              task_id=t_idx,
                                              enable_dream=True,
                                              meta_step=True,
                                              record_stats=True)
                step_loss = result.get('total_loss', result.get('loss', 0.0))

            else:
                optimizer.zero_grad()
                ref_grad = None
                if agem_module:
                    ref_grad = agem_module.get_ref_grad(device=device)
                    optimizer.zero_grad()

                # Apply Mixup (50% probability)
                do_mixup = torch.rand(1).item() < 0.5 and mixup_alpha > 0
                if do_mixup:
                    x_mix, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
                    if hat_module:
                        # HAT features must be extracted from x_mix
                        x_f = model.conv1(x_mix)
                        x_f = model.bn1(x_f); x_f = model.relu(x_f); x_f = model.maxpool(x_f)
                        x_f = model.layer1(x_f); x_f = model.layer2(x_f)
                        x_f = model.layer3(x_f); x_f = model.layer4(x_f); x_f = model.avgpool(x_f)
                        features = torch.flatten(x_f, 1)
                        masked_features = hat_module.apply_mask(features, t_idx)
                        logits = model.fc(masked_features)
                    else:
                        logits = model(x_mix.float())
                    loss = lam * F.cross_entropy(logits[:, :seen_classes], y_a, label_smoothing=label_smoothing) + \
                           (1 - lam) * F.cross_entropy(logits[:, :seen_classes], y_b, label_smoothing=label_smoothing)
                else:
                    if hat_module:
                        x_f = model.conv1(x); x_f = model.bn1(x_f); x_f = model.relu(x_f); x_f = model.maxpool(x_f)
                    if hat_module:
                        logits = model(x.float(), t_idx)
                        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
                    else:
                        logits = model(x.float())
                        loss = F.cross_entropy(logits[:, :seen_classes], y, label_smoothing=label_smoothing)

                    if hat_module:
                        loss += 0.75 * hat_module.reg_loss(t_idx)

                    if replay_buffer:
                        rx, ry = replay_buffer.get_batch(32)
                        if rx is not None:
                            r_logits = model(rx.to(device))
                            loss += F.cross_entropy(r_logits[:, :seen_classes], ry.to(device), label_smoothing=label_smoothing)

                    if der_module:
                        loss += der_module.get_loss(x, y, logits, device=device)

                    if ewc_module:
                        loss += ewc_module.penalty()
                    
                    loss.backward()

                    if agem_module:
                        ref_grad = agem_module.get_ref_grad(device=device)
                        if ref_grad is not None:
                            curr_grad = []
                            for p in model.parameters():
                                if p.grad is not None: curr_grad.append(p.grad.view(-1))
                                else: curr_grad.append(torch.zeros(p.numel(), device=device))
                            curr_grad = torch.cat(curr_grad)
                            projected = agem_module.project(curr_grad, ref_grad)
                            pointer = 0
                            for p in model.parameters():
                                p.grad = projected[pointer:pointer + p.numel()].view_as(p).clone()
                                pointer += p.numel()

                    if hat_module:
                        hat_module.mask_gradients(model)

                    optimizer.step()

                    if agem_module: agem_module.buffer.update(x, y)
                    if replay_buffer:
                        if hasattr(replay_buffer, 'update'):
                            replay_buffer.update(x, y)
                    if der_module: der_module.update(x, y, logits.detach())

                total_step_time += (time.time() - start_time) * 1000
                total_steps += 1
            print(f"\n             [DEBUG] Task {t_idx} Epoch {epoch} completed.", flush=True)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training loop failed: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise e

    avg_step_time = total_step_time / total_steps if total_steps > 0 else 0
    print(f"             [DEBUG] train_single_task for Task {t_idx} returning.", flush=True)
    return avg_step_time

def validate(model, loader, seen_classes, device, is_antara=False, hat_module=None, t_idx=0):
    """
    Validation loop. Safely extracts logits from both Antara and baseline models.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device)

            if is_antara:
                # [V28] PURE CLASS-IL: We pass task_id=None to ensure zero leakage.
                # This forces autonomous MoE routing and global head prediction.
                try:
                    logits = model.inference_step(x, task_id=None)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                except Exception:
                    # Fallback: direct forward, ensure no task-id guidance
                    out = model(x, task_id=None)
                    logits = out[0] if isinstance(out, tuple) else out
            else:
                if hat_module:
                    # HAT Inference: Must use bifurcated pass with task mask
                    x_f = model.conv1(x)
                    x_f = model.bn1(x_f)
                    x_f = model.relu(x_f)
                    x_f = model.maxpool(x_f)
                    x_f = model.layer1(x_f)
                    x_f = model.layer2(x_f)
                    x_f = model.layer3(x_f)
                    x_f = model.layer4(x_f)
                    x_f = model.avgpool(x_f)
                    features = torch.flatten(x_f, 1)
                    masked_features = hat_module.apply_mask(features, t_idx)
                    logits = model.fc(masked_features)
                else:
                    logits = model(x)

            # Class-IL Validation: No label modulo
            loss_y = y
            
            effective_classes = min(seen_classes, logits.shape[1])
            total_loss += F.cross_entropy(logits[:, :effective_classes], loss_y).item()

    return total_loss / len(loader)
