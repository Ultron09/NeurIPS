import argparse
import time
import sys
import os
import torch
from torchvision.models import resnet18

# Path setup to import from other phases
for d in ['Phase_I_Curriculum', 'Phase_II_Baselines', 'Phase_III_Metrics']:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), d))

from dataset import SplitCIFAR100, set_seed
from baselines import EWC, ExperienceReplay, AGEM
from metrics import MetricsEngine
from trainer import train_single_task
from evaluation import evaluate_suite
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig

def model_factory():
    # Use standard ResNet-18 without pre-training for benchmark purity
    return resnet18(num_classes=100)

def run_experiment(method_name, device='cuda', seed=42):
    print(f"\n[NEURIPS GAUNTLET] Executing Branch: {method_name}")
    set_seed(seed)
    curriculum = SplitCIFAR100()
    model = model_factory().to(device)
    metrics = MetricsEngine(config_name=method_name)
    total_start_time = time.time()
    task_step_times = []
    
    # Branch config
    ewc_module = None
    agem_module = None
    replay_buffer = None
    config = None
    
    if method_name == "ANTARA_FULL":
        config = AdaptiveFrameworkConfig(enable_consciousness=True, memory_type='graph', use_hierarchical_moe=True)
        model = AdaptiveFramework(model, config=config)
    elif method_name == "ANTARA_RGW_ONLY":
        config = AdaptiveFrameworkConfig(enable_consciousness=True, memory_type='none', use_hierarchical_moe=True)
        model = AdaptiveFramework(model, config=config)
    elif method_name == "ANTARA_OGD_ONLY":
        config = AdaptiveFrameworkConfig(enable_consciousness=False, memory_type='graph', use_hierarchical_moe=True)
        model = AdaptiveFramework(model, config=config)
    elif method_name == "EWC":
        ewc_module = EWC(model, lambda_factor=5000)
    elif method_name == "REPLAY":
        replay_buffer = ExperienceReplay(buffer_size=2000)
    elif method_name == "AGEM":
        agem_module = AGEM(model, buffer_size=2000)
    elif method_name == "NAIVE":
        pass
    else:
        raise ValueError("Invalid experiment method")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)

    # Main Curriculum Loop
    for t_idx in range(10):
        train_loader, val_loader, _ = curriculum.get_task(t_idx)
        
        # Unified Training Logic
        avg_step_time = train_single_task(model, train_loader, val_loader, optimizer, t_idx, 
                                          device=device, ewc_module=ewc_module, 
                                          agem_module=agem_module, replay_buffer=replay_buffer,
                                          epochs=50)
        task_step_times.append(avg_step_time)
        
        # Post-task anchoring
        if method_name == "EWC":
            ewc_module.save_task_weights(train_loader, device=device)
            
        # Unified Evaluation Autopsy
        evaluate_suite(model, curriculum, t_idx, metrics, device=device)
        
        # Checkpoint & Telemetry Update
        metrics.avg_step_time_ms = sum(task_step_times) / len(task_step_times)
        metrics.total_compute_time_sec = time.time() - total_start_time
        if device == 'cuda':
            metrics.peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            
        os.makedirs("results", exist_ok=True)
        metrics.save_results(f"results/{method_name}_metrics.json")

    metrics.generate_summary_report() # Assuming implementation in metrics.py
    metrics.plot_heatmap(f"results/{method_name}_heatmap.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_experiment(args.method, device=args.device, seed=args.seed)
