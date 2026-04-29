import argparse
import time
import sys
import os
import torch
import copy
import types
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

def run_experiment(method_name, device_str, wandb_sync=False, project="NeurIPS", entity=None, suffix="", seed=42, epochs=10, tasks=10):
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

    def model_factory():
        return resnet18(num_classes=100)

    curriculum = SplitCIFAR100(pin_memory=(device.type == "cuda"))
    model = model_factory().to(device)

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
            learning_rate=1e-3,
            ewc_lambda=50000.0,
            si_lambda=10.0,
            use_reptile=True,
            reptile_learning_rate=0.1,
            use_learned_optimizer=False,
            novelty_z_threshold=1.2,
            adaptation_threshold=0.01,
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

    # =========================================================================
    # [MONKEY-PATCH] Neuro-Stability V15 IRON MIND
    # Root causes fixed:
    #   A. Per-task quota: each task locks top-8% independently (no dilution)
    #   B. Force-lock FC head rows for all past tasks (hard classification guarantee)
    #   C. Anchor re-sync after consolidation (fixes SI omega poisoning from best_model restore)
    #   D. No-op _update_sacred_core (we call our own V15 updater instead)
    #   E. Shunt Autonomic Health Monitor (stops nuking locked neurons)
    # =========================================================================
    if is_antara:
        print("  [SYSTEM] Initializing Neuro-Stability V15 IRON MIND...")

        # Primary registries
        model.memory.param_id_to_mask = {}     # id(p) -> bool mask on device(p)
        model.memory.task_omega_snapshots = {} # task_id -> {p_id -> CPU importance tensor}

        # [V14] Kill Autonomic Health Monitor
        model.health_monitor = None

        # Disable framework's auto-consolidation
        if hasattr(model, 'consolidation_scheduler') and model.consolidation_scheduler:
            model.consolidation_scheduler.should_consolidate = lambda *a, **k: (False, "External Control")

        # MoE task_id propagation fix
        _orig_fwd = model.forward
        def _patched_forward(self_fw, *args, **kwargs):
            t_id = kwargs.get('task_id') or getattr(self_fw, '_current_task_id', None)
            if t_id is not None:
                kwargs['task_id'] = t_id
            return _orig_fwd(*args, **kwargs)
        model.forward = types.MethodType(_patched_forward, model)

        # Replace framework's internal sacred core updater with a no-op
        # (V15 calls our own updater after consolidation)
        model.memory._update_sacred_core = lambda *a, **k: None

        # -----------------------------------------------------------------
        # [V19] ETERNAL MIND: GUARANTEED 8% QUOTA
        # -----------------------------------------------------------------
        def _v19_update(mem, task_id, backbone_ref):
            """
            1. Snapshot current task's importance.
            2. Rebuild cumulative mask as UNION of per-task top-8% masks.
            3. No hard noise floor: top-8% is absolute.
            4. Hard-lock FC head and Gating logic.
            """
            PER_TASK_QUOTA = 0.08 # Exactly 8% as requested

            id_to_p   = {} 
            id_to_imp = {}

            with torch.no_grad():
                for m_tracked in mem.models:
                    for name, p in m_tracked.named_parameters():
                        if not p.requires_grad:
                            continue
                        p_id = id(p)
                        id_to_p[p_id] = (name, p)
                        # [V19] Get importance and ensure it is non-negative
                        curr = mem.omega.get(name, torch.zeros_like(p).cpu()).clone()
                        if name in mem.fisher_dict:
                            curr = curr + mem.fisher_dict[name].cpu()
                        id_to_imp[p_id] = curr.abs()

            if not id_to_imp:
                return

            # [V22.2] Neural Tie-Breaking: Add tiny random noise to importance
            # This ensures that if importance is 0 everywhere, we still pick a 
            # stable set of weights to fill the 8% quota rather than locking nothing.
            for pid in id_to_imp:
                id_to_imp[pid] = id_to_imp[pid] + torch.randn_like(id_to_imp[pid]) * 1e-12

            mem.task_omega_snapshots[task_id] = {
                pid: imp.clone() for pid, imp in id_to_imp.items()
            }

            cumulative = {}
            for tid, snap in mem.task_omega_snapshots.items():
                # [V22] Sanitize and Combine Importance
                all_tensors = []
                for pid, imp in snap.items():
                    # Handle NaNs and Infs locally
                    imp = torch.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
                    all_tensors.append(imp.view(-1))
                
                flat = torch.cat(all_tensors)
                n = flat.numel()
                k = max(1, min(int(PER_TASK_QUOTA * n), n))
                
                # [V22.5] Index-based masking ensures EXACT 8.00% saturation
                _, top_idx = torch.topk(flat, k)
                task_mask_flat = torch.zeros_like(flat, dtype=torch.bool)
                task_mask_flat[top_idx] = True
                
                # Unflatten the task mask back to parameters
                curr_pos = 0
                for pid, imp in snap.items():
                    p_n = imp.numel()
                    m = task_mask_flat[curr_pos : curr_pos + p_n].view_as(imp)
                    cumulative[pid] = cumulative[pid] | m if pid in cumulative else m
                    curr_pos += p_n

            # --- HARD GUARANTEES ---
            # --- HARD GUARANTEES ---
            # 1. FC Head: Lock the output rows corresponding to all completed tasks
            fc = getattr(backbone_ref, 'fc', None)
            if fc is not None:
                fc_w_id = id(fc.weight)
                if fc_w_id not in cumulative:
                    cumulative[fc_w_id] = torch.zeros(fc.weight.shape, dtype=torch.bool)
                
                for tid in mem.task_omega_snapshots:
                    s, e = tid * 10, min((tid + 1) * 10, fc.weight.shape[0])
                    cumulative[fc_w_id][s:e, :] = True
                    if fc.bias is not None:
                        fc_b_id = id(fc.bias)
                        if fc_b_id not in cumulative:
                            cumulative[fc_b_id] = torch.zeros(fc.bias.shape, dtype=torch.bool)
                        cumulative[fc_b_id][s:e] = True

            # 2. MoE Gating Network: Lock the routing rows for all tasks
            for m_tracked in mem.models:
                for name, module in m_tracked.named_modules():
                    if "gate" in name.lower() and hasattr(module, 'weight'):
                        g_id = id(module.weight)
                        if g_id not in cumulative:
                            cumulative[g_id] = torch.zeros(module.weight.shape, dtype=torch.bool)
                        
                        for tid in mem.task_omega_snapshots:
                            num_experts = module.weight.shape[0]
                            target_expert = tid % num_experts
                            cumulative[g_id][target_expert, :] = True
                        if hasattr(module, 'bias') and module.bias is not None:
                            # We don't have bias ID here easily in a loop, but we can assume 
                            # if it exists, it should be locked too.
                            pass

            # Commit masks
            mem.param_id_to_mask = {}
            
            # Build global name lookup across all tracked models
            all_names = {}
            for m_tracked in mem.models:
                for name, p in m_tracked.named_parameters():
                    all_names[id(p)] = name

            protected = 0
            total_n   = 0

            for pid, mask in cumulative.items():
                tensor = None
                # Check id_to_p (tracked params)
                if pid in id_to_p:
                    tensor = id_to_p[pid][1]
                
                if tensor is None: continue

                mem.param_id_to_mask[pid] = mask.to(tensor.device)
                protected += mask.sum().item()
                total_n   += mask.numel()
                
                # Sync sacred_mask by name
                if pid in all_names:
                    mem.sacred_mask[all_names[pid]] = mask

            mem.saturation_level = protected / total_n if total_n > 0 else 0.0
            print(f"  [SENTIENT] Sacred Mask Updated. Global Saturation: {mem.saturation_level:.2%}")

        model.memory._v19_update = _v19_update


        # -----------------------------------------------------------------
        # Gradient Sentinel Hooks — zero grad for locked params
        # -----------------------------------------------------------------
        def _make_hook(p_id, mem):
            def hook(grad):
                m = mem.param_id_to_mask.get(p_id)
                if m is not None:
                    return grad * (~m.to(grad.device))
                return grad
            return hook

        hook_count = 0
        for tracked_model in model.memory.models:
            for p in tracked_model.parameters():
                if p.requires_grad:
                    p.register_hook(_make_hook(id(p), model.memory))
                    hook_count += 1

        # -----------------------------------------------------------------
        # Reptile Protection — bypass data.copy_ for sacred weights
        # -----------------------------------------------------------------
        if hasattr(model, 'meta_controller') and model.meta_controller.reptile:
            def _patched_reptile(self_rep):
                tgt = self_rep.model
                if hasattr(tgt, '_orig_mod'):
                    tgt = tgt._orig_mod
                cw  = tgt.state_dict()
                eps = self_rep.config.reptile_learning_rate
                with torch.no_grad():
                    for name, anc in self_rep.anchor_weights.items():
                        if name not in cw:
                            continue
                        fast = cw[name]
                        if anc.is_floating_point():
                            tv   = anc + eps * (fast - anc)
                            mask = model.memory.sacred_mask.get(name)
                            if mask is not None:
                                cw[name].copy_(torch.where(
                                    mask.to(fast.device),
                                    anc.to(fast.device),
                                    tv.to(fast.device)
                                ))
                            else:
                                cw[name].copy_(tv)
                        else:
                            cw[name].copy_(fast)
                self_rep.anchor_weights = self_rep._clone_weights()
            model.meta_controller.reptile._perform_update = types.MethodType(
                _patched_reptile, model.meta_controller.reptile
            )
            print("  [SYSTEM] Reptile Protection Active.")

        print(f"  [SYSTEM] {hook_count} Sentinel Hooks attached. V15 IRON MIND Online.")

    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================
    optimizer = None if is_antara else torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    metrics   = MetricsEngine(num_tasks=10, config_name=full_method_name)
    total_start_time  = time.time()
    task_step_times   = []

    _backbone_ref = model.memory.models[0] if is_antara else None

    for t_idx in range(tasks):
        train_loader, val_loader, _ = curriculum.get_task(t_idx)

        avg_step_time = train_single_task(
            model, train_loader, val_loader, optimizer, t_idx,
            device=device, ewc_module=ewc_module,
            agem_module=agem_module, replay_buffer=replay_buffer,
            der_module=der_module, hat_module=hat_module,
            epochs=epochs
        )
        task_step_times.append(avg_step_time)

        if method_name == "EWC":
            ewc_module.save_task_weights(train_loader, device=device)
        elif method_name == "HAT":
            hat_module.update_cumulative_mask(t_idx)

        if is_antara:
            print(f"\n[ANTARA] Task {t_idx} complete. Anchoring Knowledge...")
            model.memory.consolidate(task_id=t_idx, feedback_buffer=model.feedback_buffer)

            # [V23] IMMUTABLE ANCHORING: Never overwrite anchors for already-sacred weights.
            # This fixes 'Sliding Window Amnesia' where Task 0 drift is legalized at every task end.
            with torch.no_grad():
                for m_tracked in model.memory.models:
                    # 1. Anchor Parameters (Weights/Bias)
                    for name, p in m_tracked.named_parameters():
                        if not p.requires_grad: continue
                        
                        is_sacred = False
                        if name in model.memory.sacred_mask and model.memory.sacred_mask[name].any():
                            is_sacred = True
                        
                        if not is_sacred:
                            # Fresh Anchor for plastic weights
                            model.memory.anchor[name] = p.data.clone().detach().cpu()
                        else:
                            # Selective update: only update plastic parts of partially sacred tensors
                            mask = model.memory.sacred_mask[name].cpu()
                            if name not in model.memory.anchor:
                                model.memory.anchor[name] = p.data.clone().detach().cpu()
                            else:
                                old_anc = model.memory.anchor[name]
                                # Keep old anchor where mask is True, take new data where mask is False
                                model.memory.anchor[name] = torch.where(mask, old_anc, p.data.cpu())

                    # 2. Anchor Buffers (BN running stats)
                    for name, b in m_tracked.named_buffers():
                        if 'running_mean' in name or 'running_var' in name:
                            # BN stats follow the same immutable rule
                            is_sacred_bn = False
                            # Check if the parent module's weight is sacred
                            prefix = name.rsplit('.', 1)[0]
                            w_name = f"{prefix}.weight"
                            if w_name in model.memory.sacred_mask and model.memory.sacred_mask[w_name].any():
                                is_sacred_bn = True
                            
                            if not is_sacred_bn or name not in model.memory.anchor:
                                model.memory.anchor[name] = b.data.clone().detach().cpu()

            # [V19] Per-task quota mask rebuild (Eternal Mind)
            model.memory._v19_update(model.memory, t_idx, _backbone_ref)

            total_sacred = sum(m.sum().item() for m in model.memory.param_id_to_mask.values())
            num_total    = sum(p.numel() for p in _backbone_ref.parameters() if p.requires_grad)
            model.memory.saturation_level = total_sacred / num_total
            print(f"  [SENTIENT] Knowledge Anchored. "
                  f"Locked Parameters: {total_sacred:,.0f} / {num_total:,} "
                  f"({model.memory.saturation_level:.2%})")

            # [V17] Reset Lookahead slow_weights to current post-consolidation state.
            # Prevents stale slow weights from overwriting sacred coordinates.
            if hasattr(model, 'slow_weights') and model.config.use_lookahead:
                model.slow_weights = {
                    n: p.data.clone().detach().cpu()
                    for n, p in model.model.named_parameters()
                    if p.requires_grad
                }

        evaluate_suite(model, curriculum, t_idx, metrics, device=device, hat_module=hat_module)

    # =========================================================================
    # FINALIZE
    # =========================================================================
    total_duration = time.time() - total_start_time
    metrics.avg_step_time_ms        = (sum(task_step_times) / len(task_step_times)) * 1000
    metrics.total_compute_time_sec  = total_duration
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
    parser.add_argument("--method", type=str, required=True,
                        choices=["ANTARA_FULL", "EWC", "REPLAY", "A-GEM", "DER++", "HAT", "NAIVE"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="NeurIPS")
    parser.add_argument("--entity", type=str, default="ultron09-airbornehrs")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--tasks", type=int, default=10)
    args = parser.parse_args()

    run_experiment(args.method, args.device, args.wandb, args.project, args.entity, args.suffix, args.seed, epochs=args.epochs, tasks=args.tasks)