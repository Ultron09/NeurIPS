import time
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_single_task(model, train_loader, val_loader, optimizer, t_idx, device='cuda',
                      ewc_module=None, agem_module=None, replay_buffer=None,
                      der_module=None, hat_module=None,
                      epochs=10, patience=3):
    """
    Unified training logic with Cognitive Bifurcation:
    - ANTARA path: delegates entirely to model.train_step() (full cognitive loop)
    - Baseline path: raw SGD loop for EWC, REPLAY, A-GEM, DER++, HAT, NAIVE
    """
    total_step_time = 0.0
    total_steps = 0

    # Dynamically detect class density (Fix for TinyImageNet/MNIST/CIFAR)
    if hasattr(train_loader.dataset, 'classes'):
        classes_per_task = len(train_loader.dataset.classes)
    else:
        # Fallback to 10 for standard benchmarks if detection fails
        classes_per_task = 10
    
    seen_classes = (t_idx + 1) * classes_per_task

    # --- BIFURCATION: Detect if model is Antara AdaptiveFramework ---
    is_antara = hasattr(model, 'train_step') and hasattr(model, 'consolidate_memory')

    for epoch in range(epochs):
        model.train()
        train_iter = tqdm(train_loader, desc=f"Task {t_idx} Epoch {epoch}")

        for x, y in train_iter:
            start_time = time.time()
            x, y = x.to(device), y.to(device)

            if is_antara:
                # [V29] MIXED-BATCH REPLAY: Combine old + new data into one step.
                # Two separate steps cause gradient tug-of-war and halve Task N learning.
                # One mixed batch lets the optimizer see a combined gradient.
                if replay_buffer and len(replay_buffer) > 0:
                    n_replay = min(max(t_idx * 8, 8), 48)  # Scale: 8→16→24→...→48
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
                # ============================================================
                # BASELINE PATH (EWC / REPLAY / A-GEM / DER++ / HAT / NAIVE)
                # ============================================================
                optimizer.zero_grad()

                # --- AGEM REFERENCE GRADIENT ---
                ref_grad = None
                if agem_module:
                    ref_grad = agem_module.get_ref_grad(device=device)
                    optimizer.zero_grad()

                # --- MAIN FORWARD PASS ---
                if hat_module:
                    # Bifurcated Forward: Extract 512-dim features for masking
                    # (Standard ResNet-18 execution order until fc)
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
                    
                    # Apply task-specific hard attention
                    masked_features = hat_module.apply_mask(features, t_idx)
                    logits = model.fc(masked_features)
                else:
                    logits = model(x.float())

                # [V27] Pure Class-IL Training (Global Cross-Entropy)
                # CHALLENGE: Distinguish current labels from all previously seen labels
                loss = F.cross_entropy(logits[:, :seen_classes], y)

                # --- HAT REGULARIZATION ---
                if hat_module:
                    # Serrà et al. (2018) sparsity penalty (c=0.75)
                    loss += 0.75 * hat_module.reg_loss(t_idx)

                # --- REPLAY INJECTION ---
                if replay_buffer:
                    rx, ry = replay_buffer.get_batch(32)
                    if rx is not None:
                        r_logits = model(rx.to(device))
                        loss += F.cross_entropy(r_logits[:, :seen_classes], ry.to(device))

                # --- DER++ LOGIT MATCHING ---
                if der_module:
                    loss += der_module.get_loss(x, y, logits, device=device)

                # --- EWC PENALTY ---
                if ewc_module:
                    loss += ewc_module.penalty()
                
                loss.backward()

                # --- AGEM PROJECTION ---
                if agem_module and ref_grad is not None:
                    curr_grad = []
                    for p in model.parameters():
                        if p.grad is not None:
                            curr_grad.append(p.grad.view(-1))
                        else:
                            curr_grad.append(torch.zeros(p.numel(), device=device))
                    curr_grad = torch.cat(curr_grad)
                    projected = agem_module.project(curr_grad, ref_grad)
                    pointer = 0
                    for p in model.parameters():
                        p.grad = projected[pointer:pointer + p.numel()].view_as(p).clone()
                        pointer += p.numel()

                # --- HAT GRADIENT MASKING ---
                if hat_module:
                    # Serrà et al. (2018): Zero gradients for weights not belonging to current task
                    hat_module.mask_gradients(model)

                optimizer.step()

                # --- POST-OPTIMIZATION BUFFER STORAGE ---
                if agem_module:
                    agem_module.buffer.update(x, y)
                if replay_buffer:
                    replay_buffer.update(x, y)
                if der_module:
                    der_module.update(x, y, logits.detach())

                step_loss = loss.item()

            total_step_time += (time.time() - start_time) * 1000  # to ms
            total_steps += 1

    avg_step_time = total_step_time / total_steps if total_steps > 0 else 0
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
