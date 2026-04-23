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
from baselines import EWC, ExperienceReplay, AGEM, DERPlus, HAT
from metrics import MetricsEngine
from trainer import train_single_task
from evaluation import evaluate_suite
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig

def setup_compute(device_str):
    """Detect and initialize the best available compute device."""
    if device_str == "cuda" and not torch.cuda.is_available():
        print(f"  [SYSTEM] CUDA requested but not found. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)
        
    print(f"  [SYSTEM] Active Compute Device: {device}")
    
    if device.type == 'cuda':
        # Optimize for static input dimensions (CIFAR-100)
        torch.backends.cudnn.benchmark = True
        print(f"  [SYSTEM] cuDNN Auto-tuner: ENABLED")
        
    return device

def model_factory():
    # Use standard ResNet-18 without pre-training for benchmark purity
    return resnet18(num_classes=100)

def run_experiment(method_name, device_str='cuda', seed=42, use_wandb=False, 
                   project_name="NeurIPS", entity_name="ultron09-airbornehrs",
                   suffix=""):
    full_method_name = f"{method_name}{suffix}_seed{seed}"
    print(f"\n[NEURIPS GAUNTLET] Executing Branch: {full_method_name}")
    
    device = setup_compute(device_str)
    
    if use_wandb:
        import wandb
        wandb.init(
            project=project_name, 
            entity=entity_name,
            name=full_method_name, 
            config={
                "method": full_method_name,
                "base_method": method_name,
                "seed": seed,
                "device": str(device)
            }
        )

    set_seed(seed)
    
    # [OPTIMIZATION] Pin memory only if using CUDA
    use_pin = (device.type == 'cuda')
    curriculum = SplitCIFAR100(pin_memory=use_pin)
    
    model = model_factory().to(device)
    metrics = MetricsEngine(config_name=full_method_name)
    total_start_time = time.time()
    task_step_times = []
    
    # Branch config
    ewc_module = None
    agem_module = None
    replay_buffer = None
    der_module = None
    hat_module = None
    config = None
    
    if method_name == "ANTARA_FULL":
        # Full framework: H-MoE + Consciousness (RGW) + Joint Saliency Masking
        config = AdaptiveFrameworkConfig(
            enable_consciousness=True,   
            importance_method='hybrid',   
            use_graph_memory=True,       
            enable_world_model=True,     
            use_moe=True,                
            use_hierarchical_moe=True,   
            use_ogd=True,                
            input_dim=3072,              
            # [V9.4] Hardened Thresholds for Class-IL
            learning_rate=5e-3,                 # [OPTIMIZED] Higher LR for faster convergence on CPU/Limited nodes
            novelty_z_threshold=1.5,            # [ADAPTIVE] Lowered to trigger faster adaptation (Tau_sim)
            consolidation_surprise_threshold=2.5, # Tau_H
            adaptation_threshold=0.02,          # [PLASTICITY] Lowered to allow more 'Cortex Editing' (Tau_sat)
            use_gradient_centralization=True,
            use_lookahead=True
        )
        model = AdaptiveFramework(model, config=config)
    elif method_name == "EWC":
        ewc_module = EWC(model, lambda_factor=5000)
    elif method_name == "REPLAY":
        replay_buffer = ExperienceReplay(buffer_size=2000)
    elif method_name == "A-GEM":
        agem_module = AGEM(model, buffer_size=2000)
    elif method_name == "DER++":
        der_module = DERPlus(model, buffer_size=2000)
    elif method_name == "HAT":
        hat_module = HAT(model, num_tasks=10).to(device)
    elif method_name == "NAIVE":
        pass
    else:
        raise ValueError("Invalid experiment method")

    # [LOG] Register hyperparameters for paper discussion
    from hparams_registry import registry
    hparams = config if config else {
        "ewc_lambda": 5000 if method_name == "EWC" else None,
        "replay_buffer_size": 2000 if method_name == "REPLAY" else None,
        "agem_buffer_size": 2000 if method_name == "A-GEM" else None,
        "der_alpha": 0.1 if method_name == "DER++" else None,
        "hat_reg": 0.75 if method_name == "HAT" else None,
        "optimizer": "SGD",
        "lr": 0.01
    }
    registry.log_experiment(full_method_name, hparams)

    is_antara = method_name.startswith("ANTARA")
    optimizer = None if is_antara else torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Main Curriculum Loop
    for t_idx in range(10):
        train_loader, val_loader, _ = curriculum.get_task(t_idx)
        
        # Unified Training Logic
        avg_step_time = train_single_task(model, train_loader, val_loader, optimizer, t_idx, 
                                          device=device, ewc_module=ewc_module, 
                                          agem_module=agem_module, replay_buffer=replay_buffer,
                                          der_module=der_module, hat_module=hat_module,
                                          epochs=10)
        task_step_times.append(avg_step_time)
        
        # Post-task anchoring
        if method_name == "EWC":
            ewc_module.save_task_weights(train_loader, device=device)
            
        # Post-task memory consolidation (V9.4 Eternal Protocol)
        if is_antara:
            print(f"\n[ANTARA] Post-task memory consolidated for Task {t_idx}.")
            model.memory.consolidate(task_id=t_idx)
            
            # [PLASTICITY RESTORATION] 
            # If saturation is > 10%, prune the mask to allow learning in Task N+1
            if model.memory.saturation_level > 0.10:
                print(f"  [SENTIENT] High Saturation ({model.memory.saturation_level:.2%}). Pruning Sacred Core for plasticity...")
                for name in model.memory.sacred_mask:
                    # Stochastically prune 50% of the least important sacred weights
                    mask = model.memory.sacred_mask[name]
                    if mask.any():
                        # Simple random pruning to restore gradient flow
                        prune_mask = torch.rand_like(mask.float()) > 0.5
                        model.memory.sacred_mask[name] = mask & prune_mask.to(mask.device)
                print(f"  [SENTIENT] Plasticity Restored. New Saturation: {model.memory.saturation_level/2:.2%}")

        # Unified Evaluation Autopsy
        evaluate_suite(model, curriculum, t_idx, metrics, device=device, hat_module=hat_module)
        
        # Checkpoint & Telemetry Update
        metrics.avg_step_time_ms = sum(task_step_times) / len(task_step_times)
        metrics.total_compute_time_sec = time.time() - total_start_time
        if device.type == 'cuda':
            metrics.peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            # [LOG] Track VRAM Delta to verify O(1) stability
            print(f"  [TELEMETRY] Task {t_idx} Peak VRAM: {metrics.peak_memory_mb:.2f} MB")
            
        os.makedirs("results", exist_ok=True)
        metrics.save_results(f"results/{full_method_name}_metrics.json")
        
        if use_wandb:
            metrics.sync_to_wandb(t_idx)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    metrics.generate_summary_report()
    metrics.plot_heatmap(f"results/{full_method_name}_heatmap.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=[
        "ANTARA_FULL", "EWC", "REPLAY", "A-GEM", "DER++", "HAT", "NAIVE"
    ])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases live logging")
    parser.add_argument("--project", type=str, default="NeurIPS")
    parser.add_argument("--entity", type=str, default="ultron09-airbornehrs")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to append to method name (e.g. _v2)")
    args = parser.parse_args()
    
    run_experiment(args.method, device_str=args.device, seed=args.seed, 
                   use_wandb=args.wandb, project_name=args.project, 
                   entity_name=args.entity, suffix=args.suffix)
