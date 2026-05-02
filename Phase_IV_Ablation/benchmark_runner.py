import os
import sys
import torch
import torch.nn as nn
import socket
import subprocess
import time
import traceback
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig

# [V28] RESOURCE TELEMETRY UTILS
try:
    import psutil
except ImportError:
    psutil = None

def get_gpu_info():
    if not torch.cuda.is_available(): return "CPU-Only"
    return torch.cuda.get_device_name(0)

def get_gpu_power():
    if not torch.cuda.is_available(): return 0.0
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return float(res.stdout.strip())
    except: return 0.0

def get_resource_usage():
    metrics = { "vram_peak_gb": 0.0, "ram_usage_gb": 0.0, "avg_power_w": 0.0 }
    if torch.cuda.is_available():
        metrics["vram_peak_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        metrics["avg_power_w"] = get_gpu_power()
    if psutil:
        metrics["ram_usage_gb"] = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    return metrics

# [V28] PATH INJECTION
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)
    sys.path.append(os.path.join(root_path, "Phase_III_Metrics"))
    sys.path.append(os.path.join(root_path, "Phase_I_Curriculum"))

from trainer import train_single_task
from dataset import SplitCIFAR100, SplitTinyImageNet

def get_node_name():
    return os.getenv("ANTARA_NODE", socket.gethostname())

def git_sync_file(filepath, message="Result Sync"):
    try:
        subprocess.run(["git", "add", filepath], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
    except Exception as e:
        print(f"[GIT_WARN] Sync failed for {filepath}: {e}")

def run_experiment(dataset_name="CIFAR100", stage_id=7, seed=42):
    node_name = get_node_name()
    method_name = { -1: "EWC", -2: "DER++", -3: "LwF", -4: "RanPAC" }.get(stage_id, f"ANTARA_S{stage_id}")
    
    # [V28] IDEMPOTENCY CHECK
    res_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(res_dir, exist_ok=True)
    filename = f"SeqN_{node_name}_{seed}_{dataset_name}_{stage_id}.txt"
    filepath = os.path.join(res_dir, filename)
    
    if os.path.exists(filepath):
        print(f"[SKIP] Result already exists: {filename}")
        return

    print(f"\n{'='*60}\nLAUNCHING: {method_name} | {dataset_name} | Seed: {seed}\n{'='*60}")
    
    # SEEDING
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(root_path, "data")
    if dataset_name == "CIFAR100": curriculum = SplitCIFAR100(root=data_path); num_classes = 100; num_tasks = 10
    else: curriculum = SplitTinyImageNet(root=data_path); num_classes = 200; num_tasks = 10
    
    from benchmark_runner import get_stage_config, model_factory, ContinualTrainer, ContinualEvaluator
    config = get_stage_config(stage_id, dataset_name)
    model = AdaptiveFramework(model_factory(dataset_name, num_classes=num_classes), config=config).to(device)
    trainer = ContinualTrainer(model, device=device); evaluator = ContinualEvaluator(model, device=device)
    
    results = []
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    for t_idx in range(num_tasks):
        train_loader, _, _ = curriculum.get_task(t_idx)
        test_loaders = [curriculum.get_task(i)[2] for i in range(t_idx + 1)]
        trainer.train_task(train_loader, t_idx, epochs=3)
        task_accuracies = [evaluator.evaluate(loader, i) for i, loader in enumerate(test_loaders)]
        avg_acc = sum(task_accuracies) / len(task_accuracies)
        print(f"Task {t_idx} AA: {avg_acc:.2%}"); results.append(avg_acc)
    
    total_time = time.time() - start_time
    usage = get_resource_usage()
    final_avg = results[-1]; bwt = (results[-1] - results[0])
    
    report = (
        f"Result File: {filename}\n"
        f"Method: {method_name} | Dataset: {dataset_name} | Seed: {seed} | Node: {node_name}\n"
        f"Avg Accuracy: {final_avg:.4f}\n"
        f"BWT: {bwt:.4f}\n"
        f"Wall-clock: {total_time/60:.2f} mins\n"
        f"Peak VRAM: {usage['vram_peak_gb']:.2f} GB\n"
        f"Mean Power: {usage['avg_power_w']:.1f} W\n"
    )
    
    with open(filepath, "w") as f: f.write(report)
    print(f"[SUCCESS] Saved to {filepath}")
    git_sync_file(filepath, f"AutoSync: {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, nargs="+", required=True)
    args = parser.parse_args()
    
    SEEDS = [42, 10, 20, 30]
    DATASETS = ["CIFAR100", "TinyImageNet"]
    
    for stage in args.stages:
        for seed in SEEDS:
            for ds in DATASETS:
                try:
                    run_experiment(dataset_name=ds, stage_id=stage, seed=seed)
                except Exception as e:
                    print(f"\n[CRITICAL_FAIL] {ds} S{stage} Seed {seed} failed. Moving to next run.")
                    print(traceback.format_exc())
                    continue
    print("\n[FINISH] All requested stages processed.")