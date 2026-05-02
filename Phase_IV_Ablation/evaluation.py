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
                    # ANTARA COGNITIVE PATH (Class-IL: No Task ID provided)
                    logits = model.inference_step(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                else:
                    # BASELINE PATH (Class-IL)
                    logits = model(x)

                # [V27] Pure Class-IL Evaluation (Global Argmax)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        accuracy = correct / total if total > 0 else 0.0
        metrics_engine.update(t_idx, eval_task_idx, accuracy)
        print(f"  Eval on Task {eval_task_idx}: {accuracy:.4f}")
