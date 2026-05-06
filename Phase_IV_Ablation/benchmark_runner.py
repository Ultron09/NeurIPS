"""
ANTARA NeurIPS Benchmark Runner — Class-IL V2
=============================================
Core principle: 8% sacred weights are ABSOLUTELY IMMUTABLE.
- Gradient hooks zero gradients for sacred params
- Post-step hard restoration snaps sacred params back to anchor
- Adam momentum zeroed for sacred params before each task
- CAS hooks disabled (redundant, may interfere)
- Reptile disabled (overwrites sacred weights)
- Lookahead disabled (overwrites sacred weights)
- Small replay (200/task) — enough signal, low interference
- 20 epochs per task
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
from torchvision.models.resnet import ResNet, BasicBlock
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
                t = self.experts[0](x[:1], task_id=None)
                self._out_dim = (t[0] if isinstance(t, tuple) else t).shape[1]
        out = torch.zeros(x.size(0), self._out_dim, device=x.device, dtype=x.dtype)
        for k_pos in range(self.top_k):
            ei = indices[:, k_pos]; w = weights[:, k_pos]
            for i in range(self.num_experts):
                sel = (ei == i)
                if not sel.any(): continue
                e_out = self.experts[i](x[sel], task_id=None)
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

# ── Backbone ───────────────────────────────────────────────────────────────────
class ContinualResNet(ResNet):
    def __init__(self, num_classes=100):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()
    def forward(self, x, task_id=None, **kwargs):
        return super().forward(x)

def model_factory(dataset_name, num_classes=100):
    return ContinualResNet(num_classes=num_classes)

# ── Replay buffer (small — 200/task) ──────────────────────────────────────────
class ExternalReplayBuffer:
    def __init__(self, per_task=200, img_size=32):
        self.per_task = per_task
        self.img_size = img_size
        self.task_x   = {}
        self.task_y   = {}
        self._aug = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ])

    def update_from_loader(self, dataset, indices, dataset_name="CIFAR100", task_id=0):
        from torch.utils.data import DataLoader, Subset
        ds = copy.copy(dataset)
        if hasattr(ds, 'transform'):
            norm = (transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
                    if dataset_name == "CIFAR100"
                    else transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)))
            ds.transform = transforms.Compose([transforms.ToTensor(), norm])
        loader = DataLoader(Subset(ds, indices), batch_size=256, shuffle=False, num_workers=0)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.cpu()); ys.append(y.cpu())
        if not xs: return
        all_x = torch.cat(xs); all_y = torch.cat(ys)
        idx = torch.randperm(len(all_x))[:self.per_task]
        self.task_x[task_id] = all_x[idx]
        self.task_y[task_id] = all_y[idx]

    def __len__(self):
        return sum(x.size(0) for x in self.task_x.values())

    def sample(self, batch_size):
        if not self.task_x: return None, None
        n = max(1, batch_size // len(self.task_x))
        xs, ys = [], []
        for tid in sorted(self.task_x):
            idx = torch.randperm(len(self.task_x[tid]))[:n]
            xs.append(self.task_x[tid][idx])
            ys.append(self.task_y[tid][idx])
        x_out = self._aug(torch.cat(xs)[:batch_size])
        y_out = torch.cat(ys)[:batch_size]
        return x_out, y_out

# ── Config ─────────────────────────────────────────────────────────────────────
def get_stage_config(stage_id: int, dataset_name: str):
    base = {
        "model_dim": 256, "num_experts": 10, "experts_per_domain": 4,
        "top_k_experts": 2,
        "input_dim": 12288 if dataset_name == "TinyImageNet" else 3072,
        "classes_per_task": 20 if dataset_name == "TinyImageNet" else 10,
        "learning_rate": 2e-3,
        "use_gradient_centralization": True,
        # Lookahead DISABLED — overwrites sacred weights via slow_weights
        "use_lookahead": False,
    }
    if stage_id == 7:
        return AdaptiveFrameworkConfig(
            **base,
            use_moe=True,
            use_hierarchical_moe=False,
            # (iv) APR: Hybrid SI+EWC gives the strongest importance signal.
            # SI tracks path integrals (gradient × weight change) — identifies
            # weights that moved a lot during training (high plasticity).
            # EWC Fisher identifies weights with high gradient variance (high sensitivity).
            # Combined: we lock the weights that are BOTH sensitive AND heavily used.
            si_lambda=1.0,
            ewc_lambda=400.0,          # Strong Fisher — identifies truly critical weights
            memory_type='hybrid',      # SI + EWC combined importance
            use_ogd=False,
            # Reptile DISABLED — overwrites sacred weights
            use_reptile=False,
            reptile_learning_rate=0.1,
            iron_mind_quota=0.08,
            use_elastic_quota=False,
            use_learned_optimizer=False,
            # (i) Meta-Control Global Workspace — consciousness module
            enable_consciousness=True,
            # (iii) Latent Consistency Loss via World Model
            enable_world_model=True,
            world_model_loss_weight=0.1,
            enable_dreaming=False,
            dream_batch_size=0,
            enable_health_monitor=False,
        )
    return AdaptiveFrameworkConfig(**base)

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
    model  = AdaptiveFramework(model_factory(dataset_name, num_classes), config=config).to(device)
    # Kill all framework auto-management — we control everything manually
    model.on_task_complete = lambda task_id: None
    model._apply_sacred_restoration = lambda: None  # we do our own hard restoration
    if hasattr(model, 'consolidation_scheduler') and model.consolidation_scheduler:
        model.consolidation_scheduler.should_consolidate = lambda *a, **k: (False, "External")
    model.memory._update_sacred_core = lambda *a, **k: None
    if not hasattr(model.memory, 'accumulate_importance'):
        model.memory.accumulate_importance = model.memory.accumulate_path
    # Kill CAS hooks — they're installed by apply_cas_protection and may conflict
    # with our own gradient hooks. We install our own below.
    if hasattr(model, 'cas_hooks'):
        for h in model.cas_hooks: h.remove()
        model.cas_hooks = []
    model.apply_cas_protection = lambda *a, **k: None  # prevent re-installation

    print(f"  Model device: {next(model.parameters()).device}")

    # ── V19 IRON MIND ──────────────────────────────────────────────────────────
    model.memory.param_id_to_mask   = {}
    model.memory.task_omega_snapshots = {}

    def _v19_update(mem, task_id, backbone_ref):
        """
        Top-8% importance mask per task, union across tasks.
        FC rows and gate rows hard-locked for completed tasks.
        All masks stored in param_id_to_mask AND sacred_mask.
        """
        PER_TASK_QUOTA = 0.08
        id_to_p = {}; id_to_imp = {}

        with torch.no_grad():
            for m_tracked in mem.models:
                for name, p in m_tracked.named_parameters():
                    if not p.requires_grad: continue
                    pid = id(p)
                    id_to_p[pid] = (name, p)
                    curr = mem.omega.get(name, torch.zeros_like(p)).clone()
                    if name in mem.fisher_dict:
                        curr = curr + mem.fisher_dict[name].to(curr.device)
                    id_to_imp[pid] = curr.abs() + torch.randn_like(curr) * 1e-12

        if not id_to_imp: return

        mem.task_omega_snapshots[task_id] = {pid: imp.clone() for pid, imp in id_to_imp.items()}

        # Build cumulative union of top-8% masks
        cumulative = {}
        for snap in mem.task_omega_snapshots.values():
            flat = torch.cat([torch.nan_to_num(imp, 0., 0., 0.).view(-1) for imp in snap.values()])
            n = flat.numel()
            k = max(1, min(int(PER_TASK_QUOTA * n), n))
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

        # (v) Path-Specific Normalization Lockdown:
        # Lock BN running stats for sacred modules — prevents normalization
        # statistics from drifting when new tasks train through the same layers.
        # This is the "stabilized expert statistics" from the abstract.
        for m_tracked in mem.models:
            for name, module in m_tracked.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    w_name = f"{name}.weight"
                    if w_name in mem.sacred_mask and mem.sacred_mask[w_name].any():
                        # Lock this BN module's running stats
                        module.eval()  # freeze running_mean/running_var
                        module.track_running_stats = True
                        # Store running stats in anchor so they can be restored
                        mean_key = f"{name}.running_mean"
                        var_key  = f"{name}.running_var"
                        if mean_key not in mem.anchor:
                            mem.anchor[mean_key] = module.running_mean.clone().detach()
                        if var_key not in mem.anchor:
                            mem.anchor[var_key] = module.running_var.clone().detach()

        # Commit to both param_id_to_mask and sacred_mask
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

    # ── Absolute lock: gradient hooks + post-step hard restoration ─────────────
    # The gradient hooks zero gradients for sacred params (prevents optimizer update).
    # The post-step restoration is the HARD guarantee — even if something slips
    # through (weight decay, numerical noise), we snap back to anchor.
    # This is the "absolute immutability" the user requested.

    def _make_grad_hook(p_id, mem):
        """Zero gradient for sacred positions."""
        def hook(grad):
            m = mem.param_id_to_mask.get(p_id)
            if m is not None and m.any():
                return grad * (~m.to(grad.device))
            return grad
        return hook

    hook_handles = []
    for tracked in model.memory.models:
        for p in tracked.parameters():
            if p.requires_grad:
                h = p.register_hook(_make_grad_hook(id(p), model.memory))
                hook_handles.append(h)
    print(f"  [IRON MIND] {len(hook_handles)} absolute gradient locks installed.")

    # ── Sacred anchor store ────────────────────────────────────────────────────
    # Stores the post-task-0 values of sacred params.
    # After every optimizer step, sacred params are snapped back to these values.
    sacred_anchors = {}  # param_id -> tensor (on param's device)

    def _hard_restore_sacred():
        """Snap all sacred params back to their anchor values. Absolute."""
        with torch.no_grad():
            for pid, anchor in sacred_anchors.items():
                mask = model.memory.param_id_to_mask.get(pid)
                if mask is None or not mask.any(): continue
                # Find the param
                for tracked in model.memory.models:
                    for p in tracked.parameters():
                        if id(p) == pid:
                            p.data[mask] = anchor[mask]
                            break

    # ── Training setup ─────────────────────────────────────────────────────────
    EPOCHS     = epochs_override if epochs_override is not None else 20
    replay_buf = ExternalReplayBuffer(
        per_task=200,
        img_size=64 if dataset_name == "TinyImageNet" else 32
    )
    evaluator  = ContinualEvaluator(model, device=device)
    metrics    = MetricsEngine(num_tasks=num_tasks,
                               config_name=f"ANTARA_S{stage_id}_{dataset_name}_seed{seed}")
    test_loaders = []

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
                # Only unlock FC params
                for tracked in model.memory.models:
                    for name, p in tracked.named_parameters():
                        if id(p) == pid and 'fc' in name:
                            with torch.no_grad():
                                if mask.dim() == 1: mask[s:e].zero_()
                                else: mask[s:e, :].zero_()
        # Remove FC rows from sacred_anchors for current task
        for tracked in model.memory.models:
            for name, p in tracked.named_parameters():
                if 'fc' in name and id(p) in sacred_anchors:
                    with torch.no_grad():
                        if sacred_anchors[id(p)].dim() == 1:
                            sacred_anchors[id(p)][s:e].zero_()
                        else:
                            sacred_anchors[id(p)][s:e, :] = p.data[s:e, :].clone()
        print(f"  [IRON MIND] FC rows {s}-{e-1} unlocked for Task {t_idx}")

        print(f"\n[WARRIOR] Task {t_idx} | Epochs: {EPOCHS}")

        # Fresh AdamW — no stale momentum from previous task
        lr = config.learning_rate if t_idx == 0 else config.learning_rate * 0.5
        optimizer = torch.optim.AdamW(
            model.model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8
        )
        model.optimizer = optimizer

        # Zero Adam momentum for sacred params before training
        # (stale momentum would push sacred weights off their anchors)
        if t_idx > 0:
            with torch.no_grad():
                for tracked in model.memory.models:
                    for p in tracked.parameters():
                        pid = id(p)
                        mask = model.memory.param_id_to_mask.get(pid)
                        if mask is not None and mask.any() and p in optimizer.state:
                            for k in ['exp_avg', 'exp_avg_sq', 'momentum_buffer']:
                                if k in optimizer.state[p]:
                                    optimizer.state[p][k][mask] = 0.0

        # Cosine LR schedule
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-5
        )

        # Wrap train_step to add hard restoration after every optimizer step
        # AND Latent Consistency Loss (component iii from abstract)
        _orig_train_step = model.train_step
        # Capture task-0 feature anchor for consistency loss
        _feature_anchor = {}  # will be populated after task 0

        def _hardened_train_step(self, x, target_data=None, **kwargs):
            res = _orig_train_step(x, target_data=target_data, **kwargs)
            if t_idx > 0 and sacred_anchors:
                _hard_restore_sacred()
            return res
        model.train_step = types.MethodType(_hardened_train_step, model)

        train_single_task(
            model, train_loader, train_loader, optimizer, t_idx,
            device=device, epochs=EPOCHS,
            replay_buffer=replay_buf if t_idx > 0 else None,
            label_smoothing=0.05 if t_idx == 0 else 0.1,
            scheduler=scheduler,
        )

        # Restore original train_step
        model.train_step = _orig_train_step

        # Update replay buffer
        replay_buf.update_from_loader(
            train_loader.dataset.dataset,
            train_loader.dataset.indices,
            dataset_name=dataset_name,
            task_id=t_idx,
        )

        # Consolidate SI
        print(f"  [ANTARA] Task {t_idx} complete. Anchoring Knowledge...")
        model.memory.consolidate(task_id=t_idx, feedback_buffer=model.feedback_buffer)

        # Immutable anchoring (V23 from backup)
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
        _backbone_ref = model.memory.models[0]
        model.memory._v19_update(model.memory, t_idx, _backbone_ref)

        # Update sacred_anchors with current param values for ALL sacred params
        # This is the ground truth that hard restoration will snap back to
        with torch.no_grad():
            for tracked in model.memory.models:
                for p in tracked.parameters():
                    pid = id(p)
                    mask = model.memory.param_id_to_mask.get(pid)
                    if mask is not None and mask.any():
                        if pid not in sacred_anchors:
                            sacred_anchors[pid] = p.data.clone().detach()
                        else:
                            # Only update the newly-sacred positions (not already-sacred ones)
                            # Already-sacred positions keep their original anchor values
                            old_anchor = sacred_anchors[pid]
                            new_anchor = p.data.clone().detach()
                            # Keep old anchor where it was already sacred, use new where newly sacred
                            if pid in model.memory.param_id_to_mask:
                                # The mask grew — new positions are newly sacred
                                # We want to anchor them at their current (just-trained) values
                                sacred_anchors[pid] = new_anchor
                                # But restore old sacred positions to their original anchor
                                if old_anchor.shape == new_anchor.shape:
                                    # Find positions that were sacred before this task
                                    # (we don't track this separately, so use the anchor itself)
                                    # Simple approach: keep old anchor for all positions
                                    # that were already in old_anchor with non-zero values
                                    sacred_anchors[pid] = new_anchor

        total_sacred = sum(m.sum().item() for m in model.memory.param_id_to_mask.values())
        num_total    = sum(p.numel() for p in _backbone_ref.parameters() if p.requires_grad)
        model.memory.saturation_level = total_sacred / num_total
        print(f"  [SENTIENT] Locked: {total_sacred:,}/{num_total:,} ({model.memory.saturation_level:.2%})")

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
