import torch
from metrics import MetricsEngine

def evaluate_suite(model, curriculum, t_idx, metrics_engine: MetricsEngine, device='cuda'):
    """
    Evaluates the model on all tasks after learning task t_idx.
    Supports Class-IL (global head) evaluation.
    """
    model.eval()
    seen_tasks = t_idx + 1
    
    for eval_task_idx in range(10):
        _, _, test_loader = curriculum.get_task(eval_task_idx)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                
                # Class-IL: Predicted class is argmax over ALL seen classes so far
                active_output_space = seen_tasks * 10
                preds = torch.argmax(logits[:, :active_output_space], dim=1)
                
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        accuracy = correct / total
        metrics_engine.update(t_idx, eval_task_idx, accuracy)
        print(f"  Eval on Task {eval_task_idx}: {accuracy:.4f}")
