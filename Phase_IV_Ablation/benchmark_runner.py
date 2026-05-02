import os
import sys
import torch
import torch.nn as nn
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig

# [V28] PATH INJECTION: Ensure sibling modules (Metrics, Data) are discoverable
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)
    sys.path.append(os.path.join(root_path, "Phase_III_Metrics"))
    sys.path.append(os.path.join(root_path, "Phase_I_Curriculum"))

from trainer import train_single_task
from evaluation import evaluate_suite
from dataset import SplitCIFAR100, SplitTinyImageNet

class ContinualTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=getattr(model.config, 'learning_rate', 2e-3))

    def train_task(self, loader, t_idx, epochs=3):
        return train_single_task(self.model, loader, loader, self.optimizer, t_idx, device=self.device, epochs=epochs)

class ContinualEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def evaluate(self, loader, t_idx):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                if hasattr(self.model, 'inference_step'):
                    logits = self.model.inference_step(x)
                else:
                    logits = self.model(x)
                if isinstance(logits, tuple): logits = logits[0]
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

def model_factory(dataset_name, num_classes=100):
    from torchvision.models import resnet18
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def get_stage_config(stage_id: int, dataset_name: str):
    base_params = {
        "model_dim": 256,
        "num_experts": 10,
        "experts_per_domain": 4,
        "top_k_experts": 2,
        "input_dim": 12288 if dataset_name == "TinyImageNet" else 3072,
        "classes_per_task": 20 if dataset_name == "TinyImageNet" else 10,
        "learning_rate": 2e-3,
        "use_gradient_centralization": True,
        "use_lookahead": True,
        "moe_temperature": 1.0,
        "moe_temp_decay": 0.90,
    }
    if stage_id == 1: return AdaptiveFrameworkConfig(**base_params, use_moe=False, si_lambda=0.0, use_reptile=False, enable_world_model=False, enable_consciousness=False)
    elif stage_id == 2: return AdaptiveFrameworkConfig(**base_params, use_moe=False, si_lambda=1.5, use_reptile=False, enable_world_model=False, enable_consciousness=False)
    elif stage_id == 3: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5, use_reptile=False, enable_world_model=False, enable_consciousness=False)
    elif stage_id == 4: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5, enable_consciousness=True, use_reptile=False, enable_world_model=False)
    elif stage_id == 5: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5, enable_consciousness=True, use_reptile=True, reptile_learning_rate=0.1, enable_world_model=False)
    elif stage_id == 6: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5, enable_consciousness=True, use_reptile=True, enable_world_model=True)
    elif stage_id == 7: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5, enable_consciousness=True, use_reptile=True, enable_world_model=True, iron_mind_quota=0.25)
    return AdaptiveFrameworkConfig(**base_params)

def run_experiment(dataset_name="CIFAR100", stage_id=7):
    print(f"\n{'='*60}\nSTARTING ANTARA NEURIPS ABLATION: STAGE {stage_id} on {dataset_name}\n{'='*60}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dataset_name == "CIFAR100":
        curriculum = SplitCIFAR100(); num_classes = 100; num_tasks = 10
    else:
        curriculum = SplitTinyImageNet(); num_classes = 200; num_tasks = 10
    config = get_stage_config(stage_id, dataset_name)
    backbone = model_factory(dataset_name, num_classes=num_classes)
    model = AdaptiveFramework(backbone, config=config).to(device)
    trainer = ContinualTrainer(model, device=device)
    evaluator = ContinualEvaluator(model, device=device)
    results = []
    for t_idx in range(num_tasks):
        train_loader, _, _ = curriculum.get_task(t_idx)
        test_loaders = [curriculum.get_task(i)[2] for i in range(t_idx + 1)]
        classes_per_task = num_classes // num_tasks
        print(f"\n--- Task {t_idx} (Classes {t_idx*classes_per_task}-{((t_idx+1)*classes_per_task)-1}) ---")
        trainer.train_task(train_loader, t_idx, epochs=3)
        task_accuracies = [evaluator.evaluate(loader, i) for i, loader in enumerate(test_loaders)]
        avg_acc = sum(task_accuracies) / len(task_accuracies)
        print(f"Task {t_idx} Complete. Avg Class-IL Accuracy: {avg_acc:.2%}")
        results.append(avg_acc)
    final_avg = results[-1]; bwt = (results[-1] - results[0])
    report = f"\nFinal NeurIPS Report (Stage {stage_id}, {dataset_name}):\n  Avg Accuracy: {final_avg:.2%}\n  BWT: {bwt:.2%}\n"
    print(report)
    
    # [V28] FAIL-SAFE LOGGING
    with open("neurips_results.txt", "a") as f:
        f.write(report + "="*30 + "\n")
        
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ANTARA Distributed Blitz Runner")
    parser.add_argument("--stages", type=int, nargs="+", required=True, help="List of Stages to run (e.g., 7 6)")
    args = parser.parse_args()
    
    # Automate Dataset Sequence for Zero-Confusion
    for ds in ["CIFAR100", "TinyImageNet"]:
        for stage in args.stages:
            run_experiment(dataset_name=ds, stage_id=stage)