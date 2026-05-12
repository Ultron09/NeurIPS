"""
ANTARA NeurIPS Benchmark Runner — V2 (Restored from backup_0.1.36.py)
======================================================================
Based on the configuration that achieved 70%+ avg acc and BWT +3.

Key differences from the broken version:
- H-MoE enabled (use_hierarchical_moe=True) — this is what the paper claims
- Reptile enabled with sacred weight protection
- Lookahead enabled with slow_weights reset after each task
- OGD enabled for gradient projection
- Standard resnet18 backbone (not custom ContinualResNet)
- 10 epochs per task (matches backup)
- No external replay buffer — framework handles internally
- V19 flat 8% quota (no inflated 30% for Task 0)
- Fisher keys harmonized (both bare name and m0_ prefix)
- Immutable anchoring (V23) — never overwrite sacred anchors
"""
import os
import sys

# [V9.8] MANDATORY: Force usage of MirrorMind library
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_path = os.path.dirname(root_path)
vendored_path = os.path.join(parent_path, "Mirror_mind")
if os.path.exists(vendored_path) and vendored_path not in sys.path:
    sys.path.insert(0, vendored_path)
    print(f"[OK] Forced MirrorMind library: {vendored_path}")

import gc
import types
import torch
import copy
import random
import socket
import time

torch.compile = lambda m, *args, **kwargs: m
import torch._dynamo
torch._dynamo.config.disable = True
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch.backends.cudnn.benchmark = True

import faulthandler
faulthandler.enable()

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import transforms
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig
import airborne_antara.moe as moe_mod

# ── CIFAR-adapted ResNet-18 ────────────────────────────────────────────────────
# Standard ResNet-18 uses 7×7 conv + maxpool which reduces 32×32 → 4×4 by layer1.
# For CIFAR-100 (32×32), we use 3×3 conv, stride 1, no maxpool — keeps spatial
# resolution high enough for BN to work with small batches.
class ContinualResNet(ResNet):
    def __init__(self, num_classes=100):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()
    def forward(self, x, task_id=None, **kwargs):
        return super().forward(x)


# ── Path setup ─────────────────────────────────────────────────────────────────
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [root_path,
          os.path.join(root_path, "Phase_III_Metrics"),
          os.path.join(root_path, "Phase_I_Curriculum"),
          os.path.dirname(os.path.abspath(__file__))]:
    if p not in sys.path:
        sys.path.append(p)

from trainer import train_single_task
from dataset import SplitCIFAR100, SplitTinyImageNet
from metrics import MetricsEngine

# ── Evaluator (pure Class-IL) ──────────────────────────────────────────────────
class ContinualEvaluator:
    def __init__(self, model, device='cuda'):
        self.model  = model
        self.device = device

    def evaluate(self, loader):
        self.model.eval()
        correct = total = 0
        # Disable Sentient Affine Modifiers during evaluation.
        # The introspection engine learns task-specific scale/shift modifiers
        # applied to every layer output via forward hooks (line 1051 in core.py:
        # inp = inp * scale + shift). After Task N trains, these modifiers are
        # calibrated for Task N's distribution — applying them to Task 0 inputs
        # amplifies Task N logits 2-3x, causing 0% Task 0 accuracy despite
        # frozen FC weights and frozen BN stats.
        saved_modifiers = getattr(self.model, 'current_modifiers', None)
        self.model.current_modifiers = None
        with torch.inference_mode():
            for x, y in loader:
                x = x.to(self.device).float()
                y = y.to(self.device)
                logits = self.model.inference_step(x)
                if isinstance(logits, tuple): logits = logits[0]
                correct += (logits.argmax(1) == y).sum().item()
                total   += y.size(0)
        self.model.current_modifiers = saved_modifiers
        return correct / total if total > 0 else 0.0

# ── Config ─────────────────────────────────────────────────────────────────────
def get_stage_config(stage_id: int, dataset_name: str):
    """
    Restored from backup_0.1.36.py ANTARA_FULL config.
    This is the configuration that achieved 70%+ avg acc and BWT +3.
    """
    base = {
        "model_dim": 256,
        "num_experts": 10,
        "top_k_experts": 2,
        "use_moe": True,
        "use_hierarchical_moe": True,   # H-MoE — what the paper claims
        "input_dim": 12288 if dataset_name == "TinyImageNet" else 3072,
        "classes_per_task": 20 if dataset_name == "TinyImageNet" else 10,
        "learning_rate": 2e-3,
        "ewc_lambda": 0.0,
        "si_lambda": 0.0,  # [V32] DISABLED: Iron Mind gradient shunts make SI redundant. SI penalty was exploding to 500+ and dominating task loss (~4.6).
        "use_reptile": True,
        "reptile_learning_rate": 0.1,
        "use_learned_optimizer": False,
        "novelty_z_threshold": 1.2,
        "adaptation_threshold": 0.05,
        "use_gradient_centralization": True,
        "use_lookahead": True,
        "use_ogd": False,  # [V32] DISABLED: Sacred mask already prevents updates to locked weights. OGD was redundant.
        "ogd_max_basis_size": 256,
        "memory_type": "si",
        "use_elastic_quota": False,
        "enable_consciousness": False,
        "enable_world_model": False,
        "world_model_loss_weight": 0.0,
        "enable_dreaming": False,
        "dream_batch_size": 0,
        "enable_health_monitor": False,
        "feedback_buffer_size": 64,
        "iron_mind_quota": 0.08,
    }
    return AdaptiveFrameworkConfig(**base)

# ── Main experiment ────────────────────────────────────────────────────────────
def run_experiment(dataset_name="CIFAR100", stage_id=7, seed=42, epochs_override=None):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    node     = socket.gethostname()
    res_dir  = os.path.join(os.getcwd(), "results"); os.makedirs(res_dir, exist_ok=True)
    out_file = os.path.join(res_dir, f"SeqN_{node}_{seed}_{dataset_name}_{stage_id}.txt")
    if os.path.exists(out_file): return

    print(f"\n{'='*60}\nLAUNCHING: ANTARA_S{stage_id} | {dataset_name} | Seed: {seed}\n{'='*60}")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(root_path, "data")

    if dataset_name == "CIFAR100":
        curriculum  = SplitCIFAR100(root=data_path, batch_size=256)
        num_classes = 100; num_tasks = 10
    else:
        curriculum  = SplitTinyImageNet(root=data_path, batch_size=256)
        num_classes = 200; num_tasks = 10

    config = get_stage_config(stage_id, dataset_name)

    # CIFAR-adapted ResNet-18 backbone — 3×3 conv, no maxpool, keeps spatial resolution
    backbone = ContinualResNet(num_classes=num_classes)
    model    = AdaptiveFramework(backbone, config=config).to(device)

    # [V31.8] ETERNAL MIND: Allow framework to handle boundaries natively.
    # We no longer disable on_task_complete because it now contains critical stability logic.
    if hasattr(model, 'consolidation_scheduler') and model.consolidation_scheduler:
        model.consolidation_scheduler.should_consolidate = lambda *a, **k: (False, "External")
    model.health_monitor = None

    print(f"  Model device: {next(model.parameters()).device}")

    # ── V19 IRON MIND ──────────────────────────────────────────────────────────
    # [V31.8] We rely on the native governance.py logic which is now identity-aware.
    # No manual _v19_update injection needed.


    # ── Gradient sentinel hooks ────────────────────────────────────────────────
    hook_count = 0
    for tracked_model in model.memory.models:
        for p in tracked_model.parameters():
            if p.requires_grad:
                def _make_hook(p_id, mem):
                    def hook(grad):
                        m = mem.param_id_to_mask.get(p_id)
                        if m is not None and m.any():
                            # [V31.11] ABSOLUTE SHUNT: Bit-perfect zeroing of sacred gradients.
                            # Prevents Adam/momentum from drifting via residuals.
                            grad.data[m] = 0.0
                            return grad
                        return grad
                    return hook
                p.register_hook(_make_hook(id(p), model.memory))
                hook_count += 1

    print(f"  [IRON MIND] {hook_count} absolute gradient locks installed.")

    # ── Reptile protection — bypass data.copy_ for sacred weights ─────────────
    if hasattr(model, 'meta_controller') and hasattr(model.meta_controller, 'reptile') \
            and model.meta_controller.reptile is not None:
        def _patched_reptile(self_rep):
            tgt = self_rep.model
            if hasattr(tgt, '_orig_mod'): tgt = tgt._orig_mod
            cw  = tgt.state_dict()
            eps = self_rep.config.reptile_learning_rate
            with torch.no_grad():
                for name, anc in self_rep.anchor_weights.items():
                    if name not in cw: continue
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

    # ── Gradient centralization — re-zero sacred grads after GC ───────────────
    _orig_gc = model._apply_gradient_centralization
    def _gc_with_sacred_zero(self_model=None):
        _orig_gc()
        with torch.no_grad():
            for tracked_model in model.memory.models:
                for p in tracked_model.parameters():
                    if p.grad is None: continue
                    mask = model.memory.param_id_to_mask.get(id(p))
                    if mask is not None and mask.any():
                        p.grad.data[mask] = 0.0
    model._apply_gradient_centralization = _gc_with_sacred_zero

    # Disable gradient noise injection (adds noise to sacred positions)
    model._steps_since_task_start = 100

    # ── Training setup ─────────────────────────────────────────────────────────
    EPOCHS     = epochs_override if epochs_override is not None else 20  # more training per task
    evaluator  = ContinualEvaluator(model, device=device)
    metrics    = MetricsEngine(num_tasks=num_tasks,
                               config_name=f"ANTARA_S{stage_id}_{dataset_name}_seed{seed}")
    test_loaders = []

    # Find all expert backbones (ContinualResNet instances) in the H-MoE hierarchy
    def _get_all_expert_backbones(moe_model):
        """Return all ContinualResNet backbones from all domains and experts."""
        backbones = []
        for name, module in moe_model.named_modules():
            if isinstance(module, ContinualResNet):
                backbones.append((name, module))
        return backbones

    _all_expert_backbones = _get_all_expert_backbones(model.memory.models[0])
    # Use first expert backbone as the reference for V19 (FC hard-lock iterates all)
    _backbone_ref = _all_expert_backbones[0][1] if _all_expert_backbones else model.memory.models[0]
    print(f"  [SYSTEM] Found {len(_all_expert_backbones)} expert backbones for Iron Mind.")

    # [V33] Define cpt before the loop — used in FC drift checks
    cpt = config.classes_per_task

    for t_idx in range(num_tasks):
        train_loader, _, test_loader = curriculum.get_task(t_idx)
        test_loaders.append(test_loader)
        train_loader.num_workers = 0; train_loader.pin_memory = True

        # [V31.11] TITANIUM ISOLATION: 
        # Manual FC unlocking removed. The framework now handles identity-aware
        # unlocking via governance.py to ensure 100% preservation of old heads.


        print(f"\n[WARRIOR] Task {t_idx} | Epochs: {EPOCHS}")

        # Disable gradient noise injection for this task
        model._steps_since_task_start = 100

        # [V31.8] ETERNAL MIND: Global BN Freeze Removed
        # Hard Expert Isolation mathematically prevents BN drift on old experts,
        # so we can safely allow the new expert to train its BN layers naturally.

        # Snapshot FC rows 0-9 AND BN running stats before training to detect any drift
        if t_idx > 0:
            _fc_pre = {}
            for exp_name, exp_bb in _all_expert_backbones:
                fc = getattr(exp_bb, 'fc', None)
                if fc is not None:
                    _fc_pre[exp_name] = fc.weight.data[:cpt].clone()
            # Snapshot BN running stats from first expert
            first_bb = _all_expert_backbones[0][1]
            for name, module in first_bb.named_modules():
                if isinstance(module, nn.BatchNorm2d) and hasattr(module, 'running_mean'):
                    _fc_pre[f"bn_{name}"] = module.running_mean.clone()

        train_single_task(
            model, train_loader, train_loader, None, t_idx,
            device=device, epochs=EPOCHS,
        )


        # Check FC drift after training
        if t_idx > 0 and _fc_pre:
            max_drift = 0.0
            for exp_name, exp_bb in _all_expert_backbones:
                fc = getattr(exp_bb, 'fc', None)
                if fc is not None and exp_name in _fc_pre:
                    drift = (fc.weight.data[:cpt] - _fc_pre[exp_name]).abs().max().item()
                    max_drift = max(max_drift, drift)
            print(f"  [FC DRIFT] Task {t_idx}: max FC rows 0-{cpt-1} drift = {max_drift:.8f} "
                  f"({'DRIFTING!' if max_drift > 1e-6 else 'HELD'})")

            # Also check BN running stats drift in first expert
            first_bb = _all_expert_backbones[0][1]
            bn_drift = 0.0
            for name, module in first_bb.named_modules():
                if isinstance(module, nn.BatchNorm2d) and hasattr(module, 'running_mean'):
                    if f"bn_{name}" in _fc_pre:
                        d = (module.running_mean - _fc_pre[f"bn_{name}"]).abs().max().item()
                        bn_drift = max(bn_drift, d)
            print(f"  [BN DRIFT] Task {t_idx}: max BN running_mean drift = {bn_drift:.6f}")

        # [V31.8] ETERNAL MIND: Standardized Weight Alignment
        # The framework now handles WA internally during on_task_complete()
        # to ensure it is applied BEFORE knowledge anchoring and cache rebuilding.
        # This eliminates the need for external post-hoc scaling.

        # ── Consolidation + Immutable Anchoring (V31.8) ────────────────────────
        # [V31.8] The framework's hardened on_task_complete handles:
        # 1. Weight Alignment (WA)
        # 2. SI/EWC Consolidation
        # 3. Mask update (via Governor)
        # 4. BN Lockdown & Cache Rebuild
        model.on_task_complete(t_idx)

        # [V31.11] TITANIUM ISOLATION:
        # Manual Fisher injection removed. Framework-native SI/EWC now operates 
        # on the expert-isolated parameters to ensure consistent importance maps.


        # Diagnostic: verify FC rows are actually locked
        cpt = config.classes_per_task
        s, e = t_idx * cpt, (t_idx + 1) * cpt
        fc_locked = 0
        for _, exp_bb in _all_expert_backbones:
            fc = getattr(exp_bb, 'fc', None)
            if fc is not None:
                pid = id(fc.weight)
                mask = model.memory.param_id_to_mask.get(pid)
                if mask is not None:
                    fc_locked += mask[:e, :].sum().item()

        print(f"  [FC CHECK] FC rows 0-{e-1} locked positions: {int(fc_locked):,} across {len(_all_expert_backbones)} experts")

        total_sacred = sum(m.sum().item() for m in model.memory.param_id_to_mask.values())
        num_total    = sum(p.numel() for m in model.memory.models
                          for p in m.parameters() if p.requires_grad)
        model.memory.saturation_level = total_sacred / num_total
        print(f"  [SENTIENT] Locked: {total_sacred:,}/{num_total:,} ({model.memory.saturation_level:.2%})")

        # V17 Reset Lookahead slow_weights AND Reptile anchor_weights after consolidation
        # Prevents stale weights from overwriting sacred coordinates on next task
        if hasattr(model, 'slow_weights') and config.use_lookahead:
            model.slow_weights = {
                n: p.data.clone().detach()
                for n, p in model.model.named_parameters()
                if p.requires_grad
            }
            print(f"  [LOOKAHEAD] Slow weights re-synced after Task {t_idx}.")

        if (hasattr(model, 'meta_controller') and
                hasattr(model.meta_controller, 'reptile') and
                model.meta_controller.reptile is not None):
            rep = model.meta_controller.reptile
            if rep.anchor_weights is not None:
                # Selective re-sync: update anchor only for NON-sacred weights.
                # Sacred weights keep their anchor from when they were first locked
                # (Reptile should not pull them back to pre-lock values).
                # Non-sacred weights get updated anchor so Reptile acts as a
                # task-transition regularizer, not a full reset.
                with torch.no_grad():
                    for name, param in model.model.named_parameters():
                        if name in rep.anchor_weights:
                            mask = model.memory.sacred_mask.get(name, None)
                            if mask is None:
                                # Fully free weight — update anchor to current value
                                rep.anchor_weights[name] = param.data.clone().detach()
                            else:
                                # Partially sacred — update only free positions
                                old_anchor = rep.anchor_weights[name]
                                new_anchor = old_anchor.clone()
                                new_anchor[~mask] = param.data[~mask]
                                rep.anchor_weights[name] = new_anchor
                print(f"  [REPTILE] Anchor selectively re-synced after Task {t_idx} (sacred positions held).")
        # Evaluate
        task_accs = []
        for i, l in enumerate(test_loaders):
            acc = evaluator.evaluate(l)
            task_accs.append(acc)
            metrics.update(t_idx, i, acc)

        # Debug: check logit distribution for Task 0 after Task 1
        if t_idx == 1:
            model.eval()
            saved_mods = getattr(model, 'current_modifiers', None)
            model.current_modifiers = None
            sample_x, sample_y = next(iter(test_loaders[0]))
            sample_x = sample_x[:8].to(device).float()
            sample_y = sample_y[:8].to(device)
            with torch.inference_mode():
                logits = model.inference_step(sample_x)
                if isinstance(logits, tuple): logits = logits[0]
            model.current_modifiers = saved_mods
            print(f"  [DEBUG] Task 0 test sample labels: {sample_y.tolist()}")
            print(f"  [DEBUG] Logits shape: {logits.shape}")
            print(f"  [DEBUG] Predicted classes: {logits.argmax(1).tolist()}")
            print(f"  [DEBUG] Logits[:, 0:10] max: {logits[:, :10].max(1).values.tolist()}")
            print(f"  [DEBUG] Logits[:, 10:20] max: {logits[:, 10:20].max(1).values.tolist()}")

        avg = sum(task_accs) / len(task_accs)
        print(f"  [LIVE] After Task {t_idx}: Avg = {avg:.2%}")
        for i, a in enumerate(task_accs):
            print(f"    Task {i}: {a:.4f}")

        torch.cuda.empty_cache(); gc.collect()

    # Final report
    acc = metrics.calculate_acc()
    bwt = metrics.calculate_bwt()
    fwt = metrics.calculate_fwt()
    print(f"\n{'='*60}")
    print(f"[NEURIPS] ANTARA_S{stage_id} | {dataset_name} | Seed {seed}")
    print(f"  ACC: {acc:.4f}  BWT: {bwt:.4f}  FWT: {fwt:.4f}")
    print(f"{'='*60}")

    mpath = os.path.join(res_dir, f"ANTARA_S{stage_id}_{dataset_name}_seed{seed}_metrics.json")
    metrics.save_results(mpath)
    metrics.plot_heatmap(os.path.join(res_dir, f"ANTARA_S{stage_id}_{dataset_name}_seed{seed}_heatmap.png"))
    with open(out_file, "w") as f:
        f.write(f"ACC: {acc:.4f} | BWT: {bwt:.4f} | FWT: {fwt:.4f}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages",  type=int, nargs="+", required=True)
    parser.add_argument("--seeds",   type=int, nargs="+", default=[42])
    parser.add_argument("--dataset", type=str, default="CIFAR100",
                        choices=["CIFAR100", "TinyImageNet"])
    parser.add_argument("--epochs",  type=int, default=None)
    args = parser.parse_args()
    for stage in args.stages:
        for seed in args.seeds:
            run_experiment(stage_id=stage, seed=seed,
                           dataset_name=args.dataset,
                           epochs_override=args.epochs)
