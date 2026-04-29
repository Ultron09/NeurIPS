import torch
from metrics import MetricsEngine


def evaluate_suite(model, curriculum, t_idx, metrics_engine: MetricsEngine, device='cuda', hat_module=None):
    """
    Evaluates the model on all tasks after learning task t_idx.
    Supports both:
    - ANTARA path: uses model.inference_step() for clean cognitive evaluation
    - Baseline path: uses raw model(x) forward pass
    Supports Class-IL (global head) evaluation.
    """
    model.eval()
    seen_tasks = t_idx + 1

    # Detect Antara framework
    is_antara = hasattr(model, 'inference_step') and hasattr(model, 'consolidate_memory')

    for eval_task_idx in range(10):
        _, _, test_loader = curriculum.get_task(eval_task_idx)
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                if is_antara:
                    # ANTARA COGNITIVE PATH
                    try:
                        # [V11] CRITICAL: Pass task_id for correct MoE routing during evaluation
                        logits = model.inference_step(x, task_id=eval_task_idx)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                    except Exception:
                        # [V11] Fallback also needs task_id propagation
                        out = model(x, task_id=eval_task_idx)
                        logits = out[0] if isinstance(out, tuple) else out
                else:
                    # BASELINE PATH
                    if hat_module:
                        # HAT EVAL: Use bifurcated pass with task-specific mask
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
                        masked_features = hat_module.apply_mask(features, eval_task_idx)
                        logits = model.fc(masked_features)
                    else:
                        logits = model(x)

                # [V25] Task-Aware Evaluation
                preds = torch.argmax(logits, dim=1)
                correct += (preds == (y % 10)).sum().item()
                total += y.size(0)

        accuracy = correct / total if total > 0 else 0.0
        metrics_engine.update(t_idx, eval_task_idx, accuracy)
        print(f"  Eval on Task {eval_task_idx}: {accuracy:.4f}")
