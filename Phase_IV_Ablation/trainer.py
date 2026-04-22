import copy
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_single_task(model, train_loader, val_loader, optimizer, t_idx, device='cuda',
                      ewc_module=None, agem_module=None, replay_buffer=None,
                      epochs=10, patience=3):
    """
    Unified training logic with Cognitive Bifurcation:
    - ANTARA path: delegates entirely to model.train_step() (full cognitive loop)
    - Baseline path: raw SGD loop for EWC, REPLAY, A-GEM, NAIVE
    """
    best_loss = float('inf')
    best_model = copy.deepcopy(model.state_dict())
    no_improve = 0
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
                # Delegates fully to AdaptiveFramework.train_step()
                # This activates: World Model, Consciousness, Introspection Engine,
                # Unified Memory (EWC/SI/Graph), OGD Projection, Lookahead,
                # Dreaming, Health Monitor, and Meta-Controller.
                # ============================================================
                try:
                    result = model.train_step(x, target_data=y,
                                              enable_dream=True,
                                              meta_step=True,
                                              record_stats=True)
                    step_loss = result.get('total_loss', result.get('loss', 0.0))
                except Exception as e:
                    # Graceful fallback: if train_step fails, log and skip
                    print(f"  [ANTARA] train_step error at T{t_idx} E{epoch}: {e}")
                    step_loss = float('inf')

            else:
                # ============================================================
                # BASELINE PATH (EWC / REPLAY / A-GEM / NAIVE)
                # Standard raw SGD loop for competitive benchmark purity.
                # ============================================================
                optimizer.zero_grad()

                # --- AGEM REFERENCE GRADIENT ---
                ref_grad = None
                if agem_module:
                    ref_grad = agem_module.get_ref_grad(device=device)
                    optimizer.zero_grad()

                # --- MAIN FORWARD PASS ---
                logits = model(x)
                loss = F.cross_entropy(logits[:, :seen_classes], y)

                # --- REPLAY INJECTION ---
                if replay_buffer:
                    rx, ry = replay_buffer.get_batch(32)
                    if rx is not None:
                        r_logits = model(rx.to(device))
                        loss += F.cross_entropy(r_logits[:, :seen_classes], ry.to(device))

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

                optimizer.step()

                # --- POST-OPTIMIZATION BUFFER STORAGE ---
                if agem_module:
                    agem_module.buffer.update(x, y)
                if replay_buffer:
                    replay_buffer.update(x, y)

                step_loss = loss.item()

            total_step_time += (time.time() - start_time) * 1000  # to ms
            total_steps += 1

        # [V9.2] Validation: Periodic check but NO early stopping
        # This keeps the training schedule fixed for scientifically sound comparison.
        val_loss = validate(model, val_loader, seen_classes, device, is_antara=is_antara)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            
    # Always restore best model weights before consolidation
    model.load_state_dict(best_model)

    # --- POST-TASK ANTARA MEMORY CONSOLIDATION ---
    # After a task is fully learned, force a hard memory consolidation.
    # This anchors the Fisher Information Matrix (EWC) and the SI path integrals
    # into permanent memory, making the next task's OGD projection bulletproof.
    if is_antara:
        try:
            model.consolidate_memory(
                feedback_buffer=model.feedback_buffer,
                current_step=getattr(model, 'step_count', 0),
                z_score=2.5,    # Force consolidation at max surprise threshold
                mode='NORMAL'
            )
            print(f"  [ANTARA] Post-task memory consolidated for Task {t_idx}.")
        except Exception as e:
            print(f"  [ANTARA] Consolidation warning: {e}")

    avg_step_time = total_step_time / total_steps if total_steps > 0 else 0
    return avg_step_time


def validate(model, loader, seen_classes, device, is_antara=False):
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
                logits = model(x)

            # Safe slicing: only score over classes seen so far
            effective_classes = min(seen_classes, logits.shape[1])
            total_loss += F.cross_entropy(logits[:, :effective_classes], y).item()

    return total_loss / len(loader)
