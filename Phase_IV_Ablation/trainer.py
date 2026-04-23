import copy
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
    best_loss = float('inf')
    best_model = copy.deepcopy(model.state_dict())
    total_step_time = 0.0
    total_steps = 0

    # Global Output Boundaries (used by baseline path only)
    seen_classes = (t_idx + 1) * 10

    # --- BIFURCATION: Detect if model is Antara AdaptiveFramework ---
    is_antara = hasattr(model, 'train_step') and hasattr(model, 'consolidate_memory')

    for epoch in range(epochs):
        model.train()
        train_iter = tqdm(train_loader, desc=f"Task {t_idx} Epoch {epoch}")

        for x, y in train_iter:
            start_time = time.time()
            x, y = x.to(device), y.to(device)

            if is_antara:
                # ============================================================
                # ANTARA COGNITIVE PATH
                # ============================================================
                try:
                    result = model.train_step(x, target_data=y,
                                              task_id=str(t_idx), 
                                              enable_dream=True,
                                              meta_step=True,
                                              record_stats=True)
                    step_loss = result.get('total_loss', result.get('loss', 0.0))
                except Exception as e:
                    print(f"  [ANTARA] train_step error at T{t_idx} E{epoch}: {e}")
                    step_loss = float('inf')

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
                    logits = model(x)

                # [V9.4] Task-Adaptive Logit Masking (NeurIPS Killshot)
                # Ensure the loss only focuses on current task classes to prevent 
                # aggressive suppression of past task knowledge.
                start_cls = t_idx * 10
                end_cls = (t_idx + 1) * 10
                task_logits = logits[:, start_cls:end_cls]
                task_y = y % 10 # Map [10-19] to [0-9]
                
                loss = F.cross_entropy(task_logits, task_y)

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
                
                # --- HAT REGULARIZATION ---
                if hat_module:
                    loss += 0.75 * hat_module.reg_loss(t_idx)

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

        # [V9.2] Validation: Periodic check but NO early stopping
        val_loss = validate(model, val_loader, seen_classes, device, is_antara=is_antara, hat_module=hat_module, t_idx=t_idx)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            
    # Always restore best model weights before consolidation
    model.load_state_dict(best_model)

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
            x, y = x.to(device), y.to(device)

            if is_antara:
                # Use inference_step for clean, diagnostic-free evaluation
                try:
                    logits = model.inference_step(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                except Exception:
                    # Fallback: direct forward, unpack tuple
                    out = model(x)
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

            # Safe slicing: only score over classes seen so far
            effective_classes = min(seen_classes, logits.shape[1])
            total_loss += F.cross_entropy(logits[:, :effective_classes], y).item()

    return total_loss / len(loader)
