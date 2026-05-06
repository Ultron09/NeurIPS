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
from torchvision.models import resnet18
from torchvision import transforms
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig
import airborne_antara.moe as moe_mod

# ── Weighted MoE dispatcher — no task_id oracle ────────────────────────────────
if hasattr(moe_mod, 'SparseMoE'):
    def _class_il_sparse_fwd(self, x, task_id=None, consciousness_state=None, *args, **kwargs):
        gate_out = self.gate(x, task_id=None, consciousness_state=consciousness_state)
        if isinstance(gate_out, tuple):
            weights, indices = gate_out
        else:
            top_k_logits, indices = torch.topk(gate_out, self.top_k, dim=1)
            weights = torch.softmax(top_k_logits, dim=1)
        if not hasattr(self, '_out_dim'):
            with torch.no_grad():
                _was_training = self.experts[0].training
                self.experts[0].eval()
                t = self.experts[0](x[:2], task_id=None)
                if _was_training:
                    self.experts[0].train()
                self._out_dim = (t[0] if isinstance(t, tuple) else t).shape[1]
        out = torch.zeros(x.size(0), self._out_dim, device=x.device, dtype=x.dtype)

        # Track expert usage
        with torch.no_grad():
            flat_indices = indices.view(-1)
            self.expert_usage.index_add_(
                0, flat_indices,
                torch.ones_like(flat_indices, dtype=self.expert_usage.dtype)
            )

        for k_pos in range(self.top_k):
            ei = indices[:, k_pos]; w = weights[:, k_pos]
            for i in range(self.num_experts):
                sel = (ei == i)
                if not sel.any(): continue
                # BN requires batch_size > 1 during training.
                # For single-sample batches, temporarily use eval mode
                # so BN uses running stats instead of batch stats.
                single_sample = self.training and sel.sum() < 2
                if single_sample:
                    self.experts[i].eval()
                e_out = self.experts[i](x[sel], task_id=None)
                if single_sample:
                    self.experts[i].train()
                if isinstance(e_out, tuple): e_out = e_out[0]
                if e_out.shape[1] == self._out_dim:
                    out[sel] += e_out * w[sel].view(-1, 1)
        return out, indices
    moe_mod.SparseMoE.forward = _class_il_sparse_fwd

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
        with torch.inference_mode():
            for x, y in loader:
                x = x.to(self.device).float()
                y = y.to(self.device)
                logits = self.model.inference_step(x)
                if isinstance(logits, tuple): logits = logits[0]
                correct += (logits.argmax(1) == y).sum().item()
                total   += y.size(0)
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
        "si_lambda": 1.0,
        "use_reptile": True,            # Reptile meta-learning
        "reptile_learning_rate": 0.1,
        "use_learned_optimizer": False,
        "novelty_z_threshold": 1.2,
        "adaptation_threshold": 0.05,
        "use_gradient_centralization": True,
        "use_lookahead": True,          # Lookahead optimizer
        "use_ogd": True,                # Gradient projection
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

    # Standard resnet18 backbone — matches backup_0.1.36.py
    backbone = resnet18(num_classes=num_classes)
    model    = AdaptiveFramework(backbone, config=config).to(device)

    # Kill framework auto-management
    model.on_task_complete = lambda task_id: None
    if hasattr(model, 'consolidation_scheduler') and model.consolidation_scheduler:
        model.consolidation_scheduler.should_consolidate = lambda *a, **k: (False, "External")
    model.memory._update_sacred_core = lambda *a, **k: None
    model.health_monitor = None

    print(f"  Model device: {next(model.parameters()).device}")

    # ── V19 IRON MIND ──────────────────────────────────────────────────────────
    model.memory.param_id_to_mask   = {}
    model.memory.task_omega_snapshots = {}

    # Per-task quota registry — each task's quota is stored so the union
    # always re-evaluates each snapshot with its ORIGINAL quota
    _task_quota_registry = {}

    def _v19_update(mem, task_id, backbone_ref):
        """
        Top-K importance mask per task, union across tasks.
        Flat 8% quota for all tasks (matches backup).
        FC rows and gate rows hard-locked for completed tasks.
        """
        PER_TASK_QUOTA = 0.08
        _task_quota_registry[task_id] = PER_TASK_QUOTA

        id_to_p = {}; id_to_imp = {}

        with torch.no_grad():
            for m_tracked in mem.models:
                for name, p in m_tracked.named_parameters():
                    if not p.requires_grad: continue
                    pid = id(p)
                    id_to_p[pid] = (name, p)
                    # Read omega with BOTH key formats (bare name and m0_ prefix)
                    curr = mem.omega.get(name, None)
                    if curr is None:
                        curr = mem.omega.get(f"m0_{name}", torch.zeros_like(p)).clone()
                    else:
                        curr = curr.clone()
                        si_key = f"m0_{name}"
                        if si_key in mem.omega:
                            curr = curr + mem.omega[si_key].to(curr.device)
                    if name in mem.fisher_dict:
                        curr = curr + mem.fisher_dict[name].to(curr.device)
                    if f"m0_{name}" in mem.fisher_dict:
                        curr = curr + mem.fisher_dict[f"m0_{name}"].to(curr.device)
                    id_to_imp[pid] = curr.abs() + torch.randn_like(curr) * 1e-12

        if not id_to_imp: return

        mem.task_omega_snapshots[task_id] = {pid: imp.clone() for pid, imp in id_to_imp.items()}

        # Build cumulative union — each snapshot uses its OWN original quota
        cumulative = {}
        for tid_snap, snap in mem.task_omega_snapshots.items():
            snap_quota = _task_quota_registry.get(tid_snap, 0.08)
            flat = torch.cat([torch.nan_to_num(imp, 0., 0., 0.).view(-1) for imp in snap.values()])
            n = flat.numel()
            k = max(1, min(int(snap_quota * n), n))
            _, top_idx = torch.topk(flat, k)
            task_flat = torch.zeros_like(flat, dtype=torch.bool)
            task_flat[top_idx] = True
            pos = 0
            for pid, imp in snap.items():
                pn = imp.numel()
                m  = task_flat[pos:pos+pn].view_as(imp)
                cumulative[pid] = cumulative[pid] | m if pid in cumulative else m
                pos += pn

        # Hard-lock FC rows for all completed tasks
        fc = getattr(backbone_ref, 'fc', None)
        if fc is not None:
            cpt = config.classes_per_task
            fc_w_id = id(fc.weight)
            if fc_w_id not in cumulative:
                cumulative[fc_w_id] = torch.zeros(fc.weight.shape, dtype=torch.bool, device=fc.weight.device)
            for tid in mem.task_omega_snapshots:
                s, e = tid * cpt, min((tid + 1) * cpt, fc.weight.shape[0])
                cumulative[fc_w_id][s:e, :] = True
            if fc.bias is not None:
                fc_b_id = id(fc.bias)
                if fc_b_id not in cumulative:
                    cumulative[fc_b_id] = torch.zeros(fc.bias.shape, dtype=torch.bool, device=fc.bias.device)
                for tid in mem.task_omega_snapshots:
                    s, e = tid * cpt, min((tid + 1) * cpt, fc.bias.shape[0])
                    cumulative[fc_b_id][s:e] = True

        # Hard-lock MoE gate routing rows for completed tasks
        for m_tracked in mem.models:
            for name, module in m_tracked.named_modules():
                if "gate" in name.lower() and hasattr(module, 'weight'):
                    g_id = id(module.weight)
                    if g_id not in cumulative:
                        cumulative[g_id] = torch.zeros(module.weight.shape, dtype=torch.bool,
                                                        device=module.weight.device)
                    for tid in mem.task_omega_snapshots:
                        num_exp = module.weight.shape[0]
                        target_exp = tid % num_exp
                        cumulative[g_id][target_exp, :] = True

        # Commit masks
        mem.param_id_to_mask = {}
        all_names = {id(p): name for m_tracked in mem.models
                     for name, p in m_tracked.named_parameters()}
        protected = total_n = 0
        for pid, mask in cumulative.items():
            if pid not in id_to_p: continue
            p = id_to_p[pid][1]
            mem.param_id_to_mask[pid] = mask.to(p.device)
            protected += mask.sum().item(); total_n += mask.numel()
            if pid in all_names:
                mem.sacred_mask[all_names[pid]] = mask.to(p.device)

        mem.saturation_level = protected / total_n if total_n > 0 else 0.0
        print(f"  [IRON MIND] Saturation: {mem.saturation_level:.2%} ({protected:,}/{total_n:,})")

    model.memory._v19_update = _v19_update

    # ── Gradient sentinel hooks ────────────────────────────────────────────────
    hook_count = 0
    for tracked_model in model.memory.models:
        for p in tracked_model.parameters():
            if p.requires_grad:
                def _make_hook(p_id, mem):
                    def hook(grad):
                        m = mem.param_id_to_mask.get(p_id)
                        if m is not None and m.any():
                            return grad * (~m.to(grad.device))
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
    EPOCHS     = epochs_override if epochs_override is not None else 10  # matches backup
    evaluator  = ContinualEvaluator(model, device=device)
    metrics    = MetricsEngine(num_tasks=num_tasks,
                               config_name=f"ANTARA_S{stage_id}_{dataset_name}_seed{seed}")
    test_loaders = []
    _backbone_ref = model.memory.models[0]

    for t_idx in range(num_tasks):
        train_loader, _, test_loader = curriculum.get_task(t_idx)
        test_loaders.append(test_loader)
        train_loader.num_workers = 0; train_loader.pin_memory = True

        # Unlock current task's FC rows before training
        cpt = config.classes_per_task
        s, e = t_idx * cpt, (t_idx + 1) * cpt
        for k, mask in model.memory.sacred_mask.items():
            if 'fc' in k.lower() and mask.dim() >= 1 and mask.shape[0] >= e:
                with torch.no_grad():
                    if mask.dim() == 1: mask[s:e].zero_()
                    else: mask[s:e, :].zero_()
        for pid, mask in model.memory.param_id_to_mask.items():
            if mask.dim() >= 1 and mask.shape[0] >= e:
                for tracked in model.memory.models:
                    for name, p in tracked.named_parameters():
                        if id(p) == pid and 'fc' in name:
                            with torch.no_grad():
                                if mask.dim() == 1: mask[s:e].zero_()
                                else: mask[s:e, :].zero_()
        print(f"  [IRON MIND] FC rows {s}-{e-1} unlocked for Task {t_idx}")

        print(f"\n[WARRIOR] Task {t_idx} | Epochs: {EPOCHS}")

        # Disable gradient noise injection for this task
        model._steps_since_task_start = 100

        train_single_task(
            model, train_loader, train_loader, None, t_idx,
            device=device, epochs=EPOCHS,
        )

        # ── Consolidation + Immutable Anchoring (V23) ──────────────────────────
        print(f"  [ANTARA] Task {t_idx} complete. Anchoring Knowledge...")
        model.memory.consolidate(task_id=t_idx, feedback_buffer=model.feedback_buffer)

        # Fast diagonal Fisher — stored with BOTH key formats
        _backbone = model.memory.models[0]
        _backbone.eval()
        _fisher_x, _fisher_y = [], []
        for _bx, _by in train_loader:
            _fisher_x.append(_bx); _fisher_y.append(_by)
            if sum(t.size(0) for t in _fisher_x) >= 256: break
        _fisher_x = torch.cat(_fisher_x)[:256].to(device).float()
        _fisher_y = torch.cat(_fisher_y)[:256].to(device)
        _backbone.zero_grad()
        with torch.enable_grad():
            _out = _backbone(_fisher_x)
            if isinstance(_out, tuple): _out = _out[0]
            F.cross_entropy(_out, _fisher_y).backward()
        with torch.no_grad():
            for name, p in _backbone.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_diag = p.grad.data.pow(2)
                    for key in [name, f"m0_{name}"]:
                        existing = model.memory.omega.get(key, torch.zeros_like(p))
                        model.memory.omega[key] = existing + fisher_diag * 400.0
        _backbone.zero_grad()
        _backbone.train()
        print(f"  [FISHER] Fast diagonal Fisher computed on {_fisher_x.size(0)} samples.")

        # V23 Immutable Anchoring — never overwrite sacred anchors
        with torch.no_grad():
            for m_tracked in model.memory.models:
                for name, p in m_tracked.named_parameters():
                    if not p.requires_grad: continue
                    is_sacred = (name in model.memory.sacred_mask and
                                 model.memory.sacred_mask[name].any())
                    if not is_sacred:
                        model.memory.anchor[name] = p.data.clone().detach()
                    else:
                        mask = model.memory.sacred_mask[name]
                        if name not in model.memory.anchor:
                            model.memory.anchor[name] = p.data.clone().detach()
                        else:
                            old = model.memory.anchor[name]
                            model.memory.anchor[name] = torch.where(
                                mask.to(p.device), old.to(p.device), p.data)
                for name, b in m_tracked.named_buffers():
                    if 'running_mean' in name or 'running_var' in name:
                        prefix = name.rsplit('.', 1)[0]
                        w_name = f"{prefix}.weight"
                        is_sacred_bn = (w_name in model.memory.sacred_mask and
                                        model.memory.sacred_mask[w_name].any())
                        if not is_sacred_bn or name not in model.memory.anchor:
                            model.memory.anchor[name] = b.data.clone().detach()

        # V19 mask rebuild
        model.memory._v19_update(model.memory, t_idx, _backbone_ref)

        total_sacred = sum(m.sum().item() for m in model.memory.param_id_to_mask.values())
        num_total    = sum(p.numel() for p in _backbone_ref.parameters() if p.requires_grad)
        model.memory.saturation_level = total_sacred / num_total
        print(f"  [SENTIENT] Locked: {total_sacred:,}/{num_total:,} ({model.memory.saturation_level:.2%})")

        # V17 Reset Lookahead slow_weights after consolidation
        # Prevents stale slow weights from overwriting sacred coordinates
        if hasattr(model, 'slow_weights') and config.use_lookahead:
            model.slow_weights = {
                n: p.data.clone().detach()
                for n, p in model.model.named_parameters()
                if p.requires_grad
            }
            print(f"  [LOOKAHEAD] Slow weights re-synced after Task {t_idx}.")

        # Evaluate
        task_accs = []
        for i, l in enumerate(test_loaders):
            acc = evaluator.evaluate(l)
            task_accs.append(acc)
            metrics.update(t_idx, i, acc)

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
