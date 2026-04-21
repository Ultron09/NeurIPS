import argparse
import time
import sys
import os
import torch
from torchvision.models import resnet18

# Path setup to import from other phases and local framework
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

def run_experiment(method_name, device='cuda', seed=42, use_wandb=False, 
                   project_name="NeurIPS", entity_name="ultron09-airbornehrs"):
    print(f"\n[NEURIPS GAUNTLET] Executing Branch: {method_name}")
    
    if use_wandb:
        import wandb
        wandb.init(
            project=project_name, 
            entity=entity_name,
            name=method_name, 
            config={
                "method": method_name,
                "seed": seed,
                "device": device
            }
        )

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
        # Full framework: H-MoE + Consciousness (RGW) + Hybrid Memory (EWC+SI) + Graph Memory (OGD)
        config = AdaptiveFrameworkConfig(
            enable_consciousness=True,   # Activates RGW (Retrograde Gating Weighting)
            memory_type='hybrid',        # Valid: 'si','ewc','hybrid','none'. Was wrongly 'graph'.
            use_graph_memory=True,       # Graph memory is a flag, NOT a memory_type value
            use_moe=True,                # REQUIRED gate: without this, use_hierarchical_moe is ignored
            use_hierarchical_moe=True,   # Activates H-MoE cortex
            use_ogd=True,                # Activates Orthogonal Gradient Descent projection
            input_dim=3072,              # [FIX] Flattened CIFAR-100 size (3*32*32) for MoE Gating
        )
        model = AdaptiveFramework(model, config=config)
    elif method_name == "ANTARA_RGW_ONLY":
        # Ablation: Only Consciousness (RGW) active. Memory/OGD disabled.
        config = AdaptiveFrameworkConfig(
            enable_consciousness=True,   # RGW active
            memory_type='none',          # Memory disabled — isolates RGW contribution
            use_moe=True,                # REQUIRED gate
            use_hierarchical_moe=True,
            use_ogd=False,               # OGD disabled — pure ablation
            input_dim=3072,              # [FIX] Match CIFAR-100
        )
        model = AdaptiveFramework(model, config=config)
    elif method_name == "ANTARA_OGD_ONLY":
        # Ablation: Only OGD+Memory active. Consciousness disabled.
        config = AdaptiveFrameworkConfig(
            enable_consciousness=False,  # RGW disabled — isolates OGD contribution
            memory_type='hybrid',        # Valid memory. Was wrongly 'graph'.
            use_graph_memory=False,
            use_moe=True,                # REQUIRED gate
            use_hierarchical_moe=True,
            use_ogd=True,                # OGD active — pure ablation
            input_dim=3072,              # [FIX] Match CIFAR-100
        )
        model = AdaptiveFramework(model, config=config)
    elif method_name == "EWC":
        ewc_module = EWC(model, lambda_factor=5000)
    elif method_name == "REPLAY":
        replay_buffer = ExperienceReplay(buffer_size=2000)
    elif method_name == "A-GEM":
        agem_module = AGEM(model, buffer_size=2000)
    elif method_name == "NAIVE":
        pass
    else:
        raise ValueError("Invalid experiment method")

    # Baselines use a shared external SGD optimizer.
    # ANTARA manages its own internal AdamW + meta-optimizer + adapter-optimizer.
    is_antara = method_name.startswith("ANTARA")
    optimizer = None if is_antara else torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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
        
        if use_wandb:
            metrics.sync_to_wandb(t_idx)

    metrics.generate_summary_report()
    metrics.plot_heatmap(f"results/{method_name}_heatmap.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=[
        "ANTARA_FULL", "ANTARA_RGW_ONLY", "ANTARA_OGD_ONLY", "EWC", "REPLAY", "A-GEM", "NAIVE"
    ])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases live logging")
    parser.add_argument("--project", type=str, default="NeurIPS")
    parser.add_argument("--entity", type=str, default="ultron09-airbornehrs")
    args = parser.parse_args()
    
    run_experiment(args.method, device=args.device, seed=args.seed, 
                   use_wandb=args.wandb, project_name=args.project, 
                   entity_name=args.entity)
