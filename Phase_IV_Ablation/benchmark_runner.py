import argparse
import time
import sys
import os
import torch
import copy
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
    if device_str == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")

def run_experiment(method_name, device_str, wandb_sync=False, project="NeurIPS", entity=None, suffix="", seed=42):
    set_seed(seed)
    device = setup_compute(device_str)
    full_method_name = f"{method_name}{suffix}_seed{seed}"
    
    print(f"\n[NEURIPS GAUNTLET] Executing Branch: {full_method_name}")
    print(f"  [SYSTEM] Active Compute Device: {device}")
    
    if wandb_sync:
        import wandb
        wandb.init(project=project, entity=entity, name=full_method_name, config={
            "method": method_name,
            "seed": seed,
            "device": str(device)
        })

    # Initialize model and components
    def model_factory():
        return resnet18(num_classes=100)
        
    curriculum = SplitCIFAR100(pin_memory=(device.type == "cuda"))
    model = model_factory().to(device)
    
    # Branch config
    ewc_module = None
    replay_buffer = None
    agem_module = None
    der_module = None
    hat_module = None
    config = None

    if method_name == "ANTARA_FULL":
        config = AdaptiveFrameworkConfig(
            model_dim=256,
            num_experts=4,
            top_k_experts=2,
            use_moe=True,
            use_hierarchical_moe=True,   
            use_ogd=True,                
            ogd_max_basis_size=256,
            input_dim=3072,              
            # [V9.4] Hardened Defense Protocol
            learning_rate=5e-3,                 
            ewc_lambda=10000.0,                 # [CRITICAL] Crank up protection
            si_lambda=10.0,                     # [CRITICAL] Online importance
            use_reptile=True,                   # [STABILITY] Manifold alignment
            reptile_learning_rate=0.1,
            use_learned_optimizer=True,         # [ADAPTATION] Meta-update
            novelty_z_threshold=1.2,            # [SENSITIVITY] Lowered for Task 1 detection
            adaptation_threshold=0.01,          # [PLASTICITY] Tightened
            use_gradient_centralization=True,
            use_lookahead=True
        )
        model = AdaptiveFramework(model, config=config, device=device)
    elif method_name == "EWC":
        ewc_module = EWC(model, lambda_factor=5000)
    elif method_name == "REPLAY":
        replay_buffer = ExperienceReplay(buffer_size=2000)
    elif method_name == "A-GEM":
        agem_module = AGEM(model, buffer_size=2000)
    elif method_name == "DER++":
        der_module = DERPlus(model, buffer_size=2000, alpha=0.1)
    elif method_name == "HAT":
        hat_module = HAT(model, num_tasks=10).to(device)
    elif method_name == "NAIVE":
        pass
    else:
        raise ValueError(f"Invalid experiment method: {method_name}")

    is_antara = method_name.startswith("ANTARA")
    
    # [MONKEY-PATCH] Fix the 100% Saturation Bug in V9.4
    if is_antara:
        def patched_update_sacred_core(self_mem, top_k_ratio=0.01):
            import torch
            all_importances = []
            with torch.no_grad():
                for model_p in self_mem.models:
                    for name, p in model_p.named_parameters():
                        if not p.requires_grad: continue
                        
                        # Hybrid Importance: SI + EWC
                        imp = self_mem.omega.get(name, torch.zeros_like(p).cpu())
                        if name in self_mem.fisher_dict:
                            imp = imp + self_mem.fisher_dict[name].cpu()
                        all_importances.append(imp.view(-1))
                
                if not all_importances: return
                
                flat_imp = torch.cat(all_importances)
                if flat_imp.max() == flat_imp.min():
                    print("  [SENTIENT] Importance is uniform. Skipping Sacred Core update.")
                    return
                
                num_total = flat_imp.numel()
                k = int(num_total * top_k_ratio)
                if k > 0:
                    threshold = torch.topk(flat_imp, k).values[-1]
                    total_sacred = 0
                    for model_p in self_mem.models:
                        for name, p in model_p.named_parameters():
                            if not p.requires_grad: continue
                            imp = self_mem.omega.get(name, torch.zeros_like(p).cpu())
                            if name in self_mem.fisher_dict:
                                imp = imp + self_mem.fisher_dict[name].cpu()
                            
                            # [V9.4 FIX] Anchor weights using >= threshold
                            new_sacred = (imp >= threshold).cpu()
                            self_mem.sacred_mask[name] = self_mem.sacred_mask.get(name, torch.zeros_like(p).cpu().bool()) | new_sacred
                            total_sacred += self_mem.sacred_mask[name].sum().item()
                    
                    self_mem.saturation_level = total_sacred / num_total
                    print(f"  [SENTIENT] Sacred Mask Updated. Global Saturation: {self_mem.saturation_level:.2%}")

        # [V9.4] Knowledge Anchoring Patch
        import types
        # [V9.4] Live Protection Registry
        # [V9.4] Live Protection Registry
        model.memory.param_id_to_mask = {}

        # Re-inject the anchoring patch with CORRECT signature
        def dynamic_id_update(self_mem, top_k_ratio=0.2):
            # 1. Run the actual anchoring logic (which I previously defined as patched_update_sacred_core)
            # But let's just implement the logic here to be safe and clean.
            
            all_importances = []
            with torch.no_grad():
                for m in self_mem.models:
                    for name, p in m.named_parameters():
                        if not p.requires_grad: continue
                        imp = self_mem.omega.get(name, torch.zeros_like(p).cpu())
                        if name in self_mem.fisher_dict:
                            imp = imp + self_mem.fisher_dict[name].cpu()
                        all_importances.append(imp.view(-1))
            
            if all_importances:
                flat_imp = torch.cat(all_importances)
                if flat_imp.numel() > 0:
                    threshold = torch.quantile(flat_imp, 1.0 - top_k_ratio)
                    # Update masks
                    for m in self_mem.models:
                        for name, p in m.named_parameters():
                            if not p.requires_grad: continue
                            imp = self_mem.omega.get(name, torch.zeros_like(p).cpu())
                            if name in self_mem.fisher_dict:
                                imp = imp + self_mem.fisher_dict[name].cpu()
                            
                            mask = (imp >= threshold).bool()
                            if name in self_mem.sacred_mask:
                                self_mem.sacred_mask[name] = self_mem.sacred_mask[name] | mask
                            else:
                                self_mem.sacred_mask[name] = mask
                            
                            # UPDATE THE ID-BASED MAP
                            if self_mem.sacred_mask[name].any():
                                self_mem.param_id_to_mask[id(p)] = self_mem.sacred_mask[name]

            # Calculate saturation
            total_params = sum(p.numel() for m in self_mem.models for p in m.parameters() if p.requires_grad)
            locked_params = sum(mask.sum().item() for mask in self_mem.sacred_mask.values())
            self_mem.saturation_level = locked_params / total_params if total_params > 0 else 0.0
            print(f"  [SENTIENT] Sacred Mask Updated. Global Saturation: {self_mem.saturation_level:.2%}")

        model.memory._update_sacred_core = types.MethodType(dynamic_id_update, model.memory)

        def get_id_sentinel_hook(p_obj, mem_obj):
            p_id = id(p_obj)
            def hook(grad):
                if p_id in mem_obj.param_id_to_mask:
                    mask = mem_obj.param_id_to_mask[p_id].to(grad.device)
                    # Surgical gradient shunting
                    return grad * (~mask)
                return grad
            return hook

        # Register hooks on ALL parameters of ALL tracked models
        hook_count = 0
        for tracked_model in model.memory.models:
            for p in tracked_model.parameters():
                if p.requires_grad:
                    p.register_hook(get_id_sentinel_hook(p, model.memory))
                    hook_count += 1
        
        print(f"  [SYSTEM] {hook_count} Dynamic Sentinel Hooks successfully attached.")

    optimizer = None if is_antara else torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    metrics = MetricsEngine(num_tasks=10, config_name=full_method_name)
    total_start_time = time.time()
    task_step_times = []

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
        elif method_name == "HAT":
            hat_module.update_cumulative_mask(t_idx)
            
        # Post-task memory consolidation (V9.4 Eternal Protocol)
        if is_antara:
            print(f"\n[ANTARA] Task {t_idx} complete. Anchoring Knowledge...")
            model.memory.consolidate(task_id=t_idx, feedback_buffer=model.feedback_buffer)
            
            # Recompute saturation
            total_sacred = sum(m.sum().item() for m in model.memory.sacred_mask.values())
            num_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.memory.saturation_level = total_sacred / num_total
            print(f"  [SENTIENT] Knowledge Anchored. Locked Parameters: {total_sacred:,} / {num_total:,} ({model.memory.saturation_level:.2%})")
            
            # [PLASTICITY RESTORATION] Safety Valve (Raised to 50% for CIFAR-100)
            if model.memory.saturation_level > 0.50:
                print(f"  [SENTIENT] High Saturation ({model.memory.saturation_level:.2%}). Restoring Plasticity...")
                for name in model.memory.sacred_mask:
                    mask = model.memory.sacred_mask[name]
                    if mask.any():
                        prune_mask = torch.rand_like(mask.float()) > 0.50
                        model.memory.sacred_mask[name] = mask & prune_mask.to(mask.device)
                
                # Recompute saturation
                total_sacred = sum(m.sum().item() for m in model.memory.sacred_mask.values())
                num_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
                model.memory.saturation_level = total_sacred / num_total
                print(f"  [SENTIENT] Plasticity Restored. New Saturation: {model.memory.saturation_level:.2%}")

        # Unified Evaluation Autopsy
        evaluate_suite(model, curriculum, t_idx, metrics, device=device, hat_module=hat_module)
        
    # Finalize
    total_duration = time.time() - total_start_time
    metrics.avg_step_time_ms = (sum(task_step_times)/len(task_step_times)) * 1000
    metrics.total_compute_time_sec = total_duration
    if device.type == "cuda":
        metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
    results_path = f"results/{full_method_name}_metrics.json"
    metrics.save_results(results_path)
    metrics.plot_heatmap(f"results/{full_method_name}_heatmap.png")
    metrics.generate_summary_report()
    
    print(f"\n[GAUNTLET COMPLETE] Method: {full_method_name}")
    print(f"  Total Duration: {total_duration/60:.2f} minutes")
    print(f"  Avg Step Time: {sum(task_step_times)/len(task_step_times):.4f}s")
    
    if wandb_sync:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["ANTARA_FULL", "EWC", "REPLAY", "A-GEM", "DER++", "HAT", "NAIVE"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="NeurIPS")
    parser.add_argument("--entity", type=str, default="ultron09-airbornehrs")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    
    run_experiment(args.method, args.device, args.wandb, args.project, args.entity, args.suffix, args.seed)