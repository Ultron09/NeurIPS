import os
import sys
import torch
import torch.nn as nn
import socket
import subprocess
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig

# [V28] PATH INJECTION
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)
    sys.path.append(os.path.join(root_path, "Phase_III_Metrics"))
    sys.path.append(os.path.join(root_path, "Phase_I_Curriculum"))

from trainer import train_single_task
from evaluation import evaluate_suite
from dataset import SplitCIFAR100, SplitTinyImageNet

def get_node_name():
    """Returns the override name if set, else fallback to system hostname."""
    return os.getenv("ANTARA_NODE", socket.gethostname())

def git_sync(message="NeurIPS Results Update"):
    node_name = get_node_name(); filename = f"neurips_results_{node_name}.txt"
    print(f"\n[GIT] Syncing {filename} from {node_name}...")
    try:
        subprocess.run(["git", "add", filename], check=True)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
        if filename in status:
            subprocess.run(["git", "commit", "-m", f"{message} from {node_name}"], check=True)
            subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=True)
            subprocess.run(["git", "push", "origin", "main"], check=True)
            print("[GIT] Sync Complete.")
    except Exception as e: print(f"[GIT] Sync Failed: {e}")

class ContinualTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model; self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=getattr(model.config, 'learning_rate', 2e-3))
    def train_task(self, loader, t_idx, epochs=3):
        return train_single_task(self.model, loader, loader, self.optimizer, t_idx, device=self.device, epochs=epochs)

class ContinualEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model; self.device = device
    def evaluate(self, loader, t_idx):
        self.model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model.inference_step(x) if hasattr(self.model, 'inference_step') else self.model(x)
                if isinstance(logits, tuple): logits = logits[0]
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item(); total += y.size(0)
        return correct / total if total > 0 else 0.0

def model_factory(dataset_name, num_classes=100):
    from torchvision.models import resnet18
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def get_stage_config(stage_id: int, dataset_name: str):
    base_params = {
        "model_dim": 256, "num_experts": 10, "experts_per_domain": 4, "top_k_experts": 2,
        "input_dim": 12288 if dataset_name == "TinyImageNet" else 3072,
        "classes_per_task": 20 if dataset_name == "TinyImageNet" else 10,
        "learning_rate": 2e-3, "use_gradient_centralization": True, "use_lookahead": True,
    }
    # --- BASELINES (Stage < 0) ---
    if stage_id == -1: return AdaptiveFrameworkConfig(**base_params, memory_type='ewc', ewc_lambda=5000, use_moe=False)
    if stage_id == -2: return AdaptiveFrameworkConfig(**base_params, memory_type='hybrid', use_prioritized_replay=True, dream_batch_size=32)
    if stage_id == -3: return AdaptiveFrameworkConfig(**base_params, memory_type='hybrid', adaptive_lambda=True)
    if stage_id == -4: return AdaptiveFrameworkConfig(**base_params, memory_type='orthogonal', use_moe=False)
    # --- ANTARA ABLATION ---
    if stage_id == 1: return AdaptiveFrameworkConfig(**base_params, use_moe=False, si_lambda=0.0)
    if stage_id == 2: return AdaptiveFrameworkConfig(**base_params, use_moe=False, si_lambda=1.5)
    if stage_id == 3: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5)
    if stage_id == 4: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5, enable_consciousness=True)
    if stage_id == 5: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5, enable_consciousness=True, use_reptile=True)
    if stage_id == 6: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5, enable_consciousness=True, use_reptile=True, enable_world_model=True)
    if stage_id == 7: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5, enable_consciousness=True, use_reptile=True, enable_world_model=True, iron_mind_quota=0.25)
    return AdaptiveFrameworkConfig(**base_params)

def run_experiment(dataset_name="CIFAR100", stage_id=7):
    method_name = { -1: "EWC", -2: "DER++", -3: "LwF", -4: "RanPAC/HOP" }.get(stage_id, f"ANTARA S{stage_id}")
    print(f"\n{'='*60}\nRUNNING {method_name} on {dataset_name}\n{'='*60}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dataset_name == "CIFAR100": curriculum = SplitCIFAR100(); num_classes = 100; num_tasks = 10
    else: curriculum = SplitTinyImageNet(); num_classes = 200; num_tasks = 10
    config = get_stage_config(stage_id, dataset_name)
    model = AdaptiveFramework(model_factory(dataset_name, num_classes=num_classes), config=config).to(device)
    trainer = ContinualTrainer(model, device=device); evaluator = ContinualEvaluator(model, device=device)
    results = []
    for t_idx in range(num_tasks):
        train_loader, _, _ = curriculum.get_task(t_idx)
        test_loaders = [curriculum.get_task(i)[2] for i in range(t_idx + 1)]
        trainer.train_task(train_loader, t_idx, epochs=3)
        task_accuracies = [evaluator.evaluate(loader, i) for i, loader in enumerate(test_loaders)]
        avg_acc = sum(task_accuracies) / len(task_accuracies)
        print(f"Task {t_idx} Complete. AA: {avg_acc:.2%}"); results.append(avg_acc)
    final_avg = results[-1]; bwt = (results[-1] - results[0]); node_name = get_node_name()
    report = f"\nFinal Report ({method_name}, {dataset_name}, Node: {node_name}):\n  Avg Accuracy: {final_avg:.2%}\n  BWT: {bwt:.2%}\n"
    print(report)
    with open(f"neurips_results_{node_name}.txt", "a") as f: f.write(report + "="*30 + "\n")
    git_sync(f"Completed {method_name} on {dataset_name}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--stages", type=int, nargs="+", required=True)
    args = parser.parse_args()
    try:
        for ds in ["CIFAR100", "TinyImageNet"]:
            for stage in args.stages: run_experiment(dataset_name=ds, stage_id=stage)
    except KeyboardInterrupt: print("\n[ALERT] Interrupted. Syncing...")
    finally: git_sync("Final Lifecycle Sync")