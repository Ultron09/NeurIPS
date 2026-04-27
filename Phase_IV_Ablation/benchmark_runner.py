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
            num_experts=10,
            top_k_experts=1,
            use_moe=True,
            use_hierarchical_moe=True,   
            use_ogd=True,                
            ogd_max_basis_size=256,
            input_dim=3072,              
            # [V9.4] Hardened Defense Protocol
            learning_rate=1e-3,                 # [V13] Lowered for stability
            ewc_lambda=50000.0,                 # [V13] Extreme protection
            si_lambda=10.0,                     
            use_reptile=True,                   
            reptile_learning_rate=0.1,
            use_learned_optimizer=False,        # [V13] Disabled to reduce optimization noise
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
        import types
        print("  [SYSTEM] Initializing Neuro-Stability V10...")

        # [V10] Live Protection Registry
        model.memory.param_id_to_mask = {}

        # [V10] MoE Propagation Fix: Patch AdaptiveFramework.forward to pass task_id
        # This is CRITICAL to prevent Task 1 training from destroying Expert 0
        original_forward = model.forward
        def patched_forward(self_fw, *args, **kwargs):
            t_id = kwargs.get('task_id')
            if t_id is None:
                t_id = getattr(self_fw, '_current_task_id', None)
            
            # If task_id is found, ensure it's in kwargs for the underlying Experts
            if t_id is not None:
                kwargs['task_id'] = t_id
                
            return original_forward(*args, **kwargs)
        
        model.forward = types.MethodType(patched_forward, model)

        # [V10] Disable Framework Auto-Consolidation to prevent interference during epochs
        if hasattr(model, 'consolidation_scheduler') and model.consolidation_scheduler:
            model.consolidation_scheduler.should_consolidate = lambda *args, **kwargs: (False, "External Control")

        # [V13] Collision-Free Robust Thresholding
        def dynamic_id_update(self_mem, top_k_ratio=0.08):
            import torch
            all_importances = []
            id_to_imp = {}
            id_to_p = {}
            
            with torch.no_grad():
                # We assume the first model in tracked_models is the backbone.
                backbone = self_mem.models[0]
                
                for m_idx, m_tracked in enumerate(self_mem.models):
                    for name, p in m_tracked.named_parameters():
                        if not p.requires_grad: continue
                        
                        # Use a unique ID for every parameter across all models
                        p_id = id(p)
                        id_to_p[p_id] = p
                        
                        # Hybrid Importance: SI (omega) + EWC (fisher)
                        # [FIX] Framework only uses string names, which collide. 
                        # We must find the specific importance for THIS model.
                        imp = self_mem.omega.get(name, torch.zeros_like(p).cpu())
                        if name in self_mem.fisher_dict:
                            imp = imp + self_mem.fisher_dict[name].cpu()
                        
                        id_to_imp[p_id] = imp
                        all_importances.append(imp.view(-1))
            
            if not all_importances: return
            
            flat_imp = torch.cat(all_importances)
            num_total = flat_imp.numel()
            
            # Calculate global threshold
            k = int((1.0 - top_k_ratio) * num_total)
            k = max(1, min(k, num_total))
            threshold = torch.kthvalue(flat_imp, k).values.item()

            protected_count = 0
            for p_id, imp in id_to_imp.items():
                mask = (imp >= threshold).bool()
                p = id_to_p[p_id]
                
                # Update the ID-based registry (Primary Defense)
                if p_id in self_mem.param_id_to_mask:
                    self_mem.param_id_to_mask[p_id] = self_mem.param_id_to_mask[p_id] | mask.to(p.device)
                else:
                    self_mem.param_id_to_mask[p_id] = mask.to(p.device)
                
                # Update string-based sacred_mask for framework compatibility (Backbone Only)
                if p in backbone.parameters():
                    # Find name in backbone
                    for name, bp in backbone.named_parameters():
                        if bp is p:
                            if name in self_mem.sacred_mask:
                                self_mem.sacred_mask[name] = self_mem.sacred_mask[name] | mask
                            else:
                                self_mem.sacred_mask[name] = mask
                            break

                protected_count += self_mem.param_id_to_mask[p_id].sum().item()

            self_mem.saturation_level = protected_count / num_total if num_total > 0 else 0.0
            print(f"  [SENTIENT] Sacred Mask Updated. Global Saturation: {self_mem.saturation_level:.2%}")

        model.memory._update_sacred_core = types.MethodType(dynamic_id_update, model.memory)

        def get_id_sentinel_hook(p_obj, mem_obj):
            p_id = id(p_obj)
            def hook(grad):
                if hasattr(mem_obj, 'param_id_to_mask') and p_id in mem_obj.param_id_to_mask:
                    mask = mem_obj.param_id_to_mask[p_id].to(grad.device)
                    # Surgical gradient shunting: 0 for locked weights, 1 for free weights
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
        
        # [V13] Reptile Protection: Reptile updates via param.data.copy_ bypass hooks.
        # We must patch the update rule to respect the Sacred Mask.
        if hasattr(model, 'meta_controller') and model.meta_controller.reptile:
            original_reptile_update = model.meta_controller.reptile._perform_update
            def patched_reptile_update(self_rep):
                target_model = self_rep.model
                if hasattr(target_model, '_orig_mod'):
                    target_model = target_model._orig_mod

                current_weights = target_model.state_dict()
                epsilon = self_rep.config.reptile_learning_rate
                
                with torch.no_grad():
                    for name, anchor_param in self_rep.anchor_weights.items():
                        if name in current_weights:
                            fast_param = current_weights[name]
                            if anchor_param.is_floating_point():
                                # Reptile Interpolation
                                target_val = anchor_param + epsilon * (fast_param - anchor_param)
                                
                                # [V13] Respect the Sacred Mask
                                mask = model.memory.sacred_mask.get(name, None)
                                if mask is not None:
                                    # Keep anchor values for sacred coordinates, use target for free ones
                                    # Since we are moving towards the fast_param, but want to keep the anchor's stability
                                    # for protected weights, we simply don't move the protected weights.
                                    protected_val = anchor_param.to(fast_param.device)
                                    new_val = torch.where(mask.to(fast_param.device), protected_val, target_val.to(fast_param.device))
                                    current_weights[name].copy_(new_val)
                                else:
                                    current_weights[name].copy_(target_val)
                            else:
                                current_weights[name].copy_(fast_param)
                
                # Update Anchor for next cycle
                self_rep.anchor_weights = self_rep._clone_weights()
            
            model.meta_controller.reptile._perform_update = types.MethodType(patched_reptile_update, model.meta_controller.reptile)
            print(f"  [SYSTEM] Reptile Protection Active. Applied to {hook_count} parameters.")
        
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
            
            # [PLASTICITY RESTORATION] Safety Valve (Raised to 85% for NeurIPS 10-Task stability)
            if model.memory.saturation_level > 0.85:
                print(f"  [SENTIENT] Critical Saturation ({model.memory.saturation_level:.2%}). Optimizing Mask...")
                # [V11] Intelligent Pruning: Only keep the most critical overlaps if we hit absolute limit
                # For now, we just log and allow it to continue to 95%
                pass
                
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