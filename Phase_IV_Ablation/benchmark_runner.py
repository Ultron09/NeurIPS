"""
ANTARA NeurIPS Benchmark Runner — Class-IL
==========================================
Architecture: backup_0.1.36.py V19 IRON MIND, adapted for strict Class-IL.

Key design decisions vs backup:
- Evaluation: global argmax over all 100 classes (no task-ID oracle)
- Training: identical to backup — gradient hooks zero sacred grads, that's it
- SI lambda=1.0 (same as backup) — protection is via hard mask, not SI penalty
- Anchors stored with UNPREFIXED keys (same as backup's _v19_update)
- No post-step restoration — gradient zeroing is sufficient
- Lookahead reset after each task (same as backup)
- External replay buffer for class-IL stability
"""
import os
import sys
import gc
import types
import torch
import copy
import random
import socket
import subprocess
import time

torch.compile = lambda m, *args, **kwargs: m
import torch._dynamo
torch._dynamo.config.disable = True
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch.backends.cudnn.benchmark = True   # re-enable for A100 speed

import faulthandler
faulthandler.enable()

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import transforms
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig
import airborne_antara.moe as moe_mod

# ── Weighted MoE dispatcher (class-IL: no task_id oracle in routing) ──────────
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
            ei = indices[:, k_pos]
            w  = weights[:, k_pos]
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
        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()
    def forward(self, x, task_id=None, **kwargs):
        return super().forward(x)

def model_factory(dataset_name, num_classes=100):
    return ContinualResNet(num_classes=num_classes)

# ── Replay buffer ──────────────────────────────────────────────────────────────
class ExternalReplayBuffer:
    """Balanced per-task reservoir buffer."""
    def __init__(self, per_task=2000, img_size=32):
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
        "use_lookahead": True,
    }
    if stage_id == 7:
        return AdaptiveFrameworkConfig(
            **base,
            use_moe=True,
            use_hierarchical_moe=False,
            # SI lambda=1.0 — same as backup. Protection is via hard gradient mask,
            # not SI penalty. SI penalty at 1.0 is negligible and won't cause divergence.
            si_lambda=1.0,
            ewc_lambda=0.0,
            memory_type='si',
            use_ogd=False,
            ogd_max_basis_size=256,
            novelty_z_threshold=1.2,
            adaptation_threshold=0.05,
            enable_consciousness=False,   # disabled for speed and stability
            use_reptile=True,             # kept — backup used it, we patch it
            reptile_learning_rate=0.1,
            iron_mind_quota=0.08,
            use_elastic_quota=False,
            use_learned_optimizer=False,
            enable_world_model=False,
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
    from torch.utils.data import DataLoader
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
    model.on_task_complete = lambda task_id: None
    print(f"  Model device: {next(model.parameters()).device}")

    # ── V19 IRON MIND (exact port from backup_0.1.36.py) ──────────────────────
    model.memory.param_id_to_mask   = {}
    model.memory.task_omega_snapshots = {}
    if hasattr(model, 'consolidation_scheduler') and model.consolidation_scheduler:
        model.consolidation_scheduler.should_consolidate = lambda *a, **k: (False, "External")
    model.memory._update_sacred_core = lambda *a, **k: None
    if not hasattr(model.memory, 'accumulate_importance'):
        model.memory.accumulate_importance = model.memory.accumulate_path

    # Disable package's own restoration — we use gradient hooks only (backup style)
    model._apply_sacred_restoration = lambda: None
    # Disable lookahead interference — we reset slow_weights after each task instead
    # (same as backup's V17 reset)

    def _v19_update(mem, task_id, backbone_ref):
        """
        Exact V19 from backup_0.1.36.py.
        Reads omega with UNPREFIXED keys (same as backup).
        Locks FC rows for all completed tasks.
        """
        PER_TASK_QUOTA = 0.08
        id_to_p   = {}
        id_to_imp = {}

        with torch.no_grad():
            for m_tracked in mem.models:
                for name, p in m_tracked.named_parameters():
                    if not p.requires_grad: continue
                    pid = id(p)
                    id_to_p[pid] = (name, p)
                    # Read omega with UNPREFIXED key (backup style)
                    curr = mem.omega.get(name, torch.zeros_like(p)).clone()
                    if name in mem.fisher_dict:
                        curr = curr + mem.fisher_dict[name].to(curr.device)
                    id_to_imp[pid] = curr.abs()

        if not id_to_imp: return

        # Tie-breaking noise
        for pid in id_to_imp:
            id_to_imp[pid] = id_to_imp[pid] + torch.randn_like(id_to_imp[pid]) * 1e-12

        mem.task_omega_snapshots[task_id] = {pid: imp.clone() for pid, imp in id_to_imp.items()}

        cumulative = {}
        for snap in mem.task_omega_snapshots.values():
            all_t = [torch.nan_to_num(imp, 0., 0., 0.).view(-1) for imp in snap.values()]
            flat  = torch.cat(all_t)
            n     = flat.numel()
            k     = max(1, min(int(PER_TASK_QUOTA * n), n))
            _, top_idx = torch.topk(flat, k)
            task_flat  = torch.zeros_like(flat, dtype=torch.bool)
            task_flat[top_idx] = True
            pos = 0
            for pid, imp in snap.items():
                pn = imp.numel()
                m  = task_flat[pos:pos+pn].view_as(imp)
                cumulative[pid] = cumulative[pid] | m if pid in cumulative else m
                pos += pn

        # Hard-lock FC rows for all completed tasks (including current)
        fc = getattr(backbone_ref, 'fc', None)
        if fc is not None:
            fc_w_id = id(fc.weight)
            if fc_w_id not in cumulative:
                cumulative[fc_w_id] = torch.zeros(fc.weight.shape, dtype=torch.bool, device=fc.weight.device)
            cpt = config.classes_per_task
            for tid in mem.task_omega_snapshots:
                s, e = tid * cpt, min((tid + 1) * cpt, fc.weight.shape[0])
                cumulative[fc_w_id][s:e, :] = True
                if fc.bias is not None:
                    fc_b_id = id(fc.bias)
                    if fc_b_id not in cumulative:
                        cumulative[fc_b_id] = torch.zeros(fc.bias.shape, dtype=torch.bool, device=fc.bias.device)
                    cumulative[fc_b_id][s:e] = True

        # Commit
        mem.param_id_to_mask = {}
        all_names = {}
        for m_tracked in mem.models:
            for name, p in m_tracked.named_parameters():
                all_names[id(p)] = name

        protected = total_n = 0
        for pid, mask in cumulative.items():
            if pid not in id_to_p: continue
            tensor = id_to_p[pid][1]
            mem.param_id_to_mask[pid] = mask.to(tensor.device)
            protected += mask.sum().item()
            total_n   += mask.numel()
            if pid in all_names:
                mem.sacred_mask[all_names[pid]] = mask

        mem.saturation_level = protected / total_n if total_n > 0 else 0.0
        print(f"  [IRON MIND] Saturation: {mem.saturation_level:.2%} ({protected:,}/{total_n:,})")

    model.memory._v19_update = _v19_update

    # ── Gradient sentinel hooks (exact from backup) ────────────────────────────
    def _make_hook(p_id, mem):
        def hook(grad):
            m = mem.param_id_to_mask.get(p_id)
            if m is not None:
                return grad * (~m.to(grad.device))
            return grad
        return hook

    hook_count = 0
    for tracked in model.memory.models:
        for p in tracked.parameters():
            if p.requires_grad:
                p.register_hook(_make_hook(id(p), model.memory))
                hook_count += 1
    print(f"  [IRON MIND] {hook_count} gradient sentinel hooks installed.")

    # ── Reptile protection (exact from backup) ─────────────────────────────────
    if hasattr(model, 'meta_controller') and model.meta_controller and \
       hasattr(model.meta_controller, 'reptile') and model.meta_controller.reptile:
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
                            cw[name].copy_(torch.where(mask.to(fast.device),
                                                        anc.to(fast.device),
                                                        tv.to(fast.device)))
                        else:
                            cw[name].copy_(tv)
                    else:
                        cw[name].copy_(fast)
            self_rep.anchor_weights = self_rep._clone_weights()
        model.meta_controller.reptile._perform_update = types.MethodType(
            _patched_reptile, model.meta_controller.reptile)
        print("  [IRON MIND] Reptile protection active.")

    # ── Training setup ─────────────────────────────────────────────────────────
    EPOCHS     = epochs_override if epochs_override is not None else 20
    replay_buf = ExternalReplayBuffer(
        per_task=2000,
        img_size=64 if dataset_name == "TinyImageNet" else 32
    )
    evaluator  = ContinualEvaluator(model, device=device)
    metrics    = MetricsEngine(num_tasks=num_tasks,
                               config_name=f"ANTARA_S{stage_id}_{dataset_name}_seed{seed}")
    test_loaders = []

    for t_idx in range(num_tasks):
        train_loader, _, test_loader = curriculum.get_task(t_idx)
        test_loaders.append(test_loader)
        train_loader.num_workers = 0
        train_loader.pin_memory  = True

        # ── Before training: unlock current task's FC rows ─────────────────────
        # The governor locks rows 0..(t_idx-1)*cpt. Rows t_idx*cpt+ are free.
        # This is a safety check — ensures no stale mask blocks new task's head.
        cpt = config.classes_per_task
        s, e = t_idx * cpt, (t_idx + 1) * cpt
        for k, mask in model.memory.sacred_mask.items():
            if 'fc' in k.lower() and mask.dim() >= 1 and mask.shape[0] >= e:
                with torch.no_grad():
                    if mask.dim() == 1:
                        mask[s:e].zero_()
                    else:
                        mask[s:e, :].zero_()
        # Also unlock in param_id_to_mask
        for pid, mask in model.memory.param_id_to_mask.items():
            if pid in {id(p) for _, p in model.memory.models[0].named_parameters()
                       if 'fc' in _ and p.dim() >= 1 and p.shape[0] >= e}:
                with torch.no_grad():
                    if mask.dim() == 1: mask[s:e].zero_()
                    else: mask[s:e, :].zero_()
        print(f"  [IRON MIND] FC rows {s}-{e-1} unlocked for Task {t_idx}")

        print(f"\n[WARRIOR] Task {t_idx} | Epochs: {EPOCHS}")

        # Fresh optimizer for each task (same as backup — no stale momentum)
        lr = config.learning_rate if t_idx == 0 else config.learning_rate * 0.5
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr,
                                      weight_decay=1e-4, eps=1e-8)
        model.optimizer = optimizer

        train_single_task(
            model, train_loader, train_loader, optimizer, t_idx,
            device=device, epochs=EPOCHS,
            replay_buffer=replay_buf if t_idx > 0 else None,
            label_smoothing=0.05 if t_idx == 0 else 0.1,
        )

        # ── Post-task: update replay buffer ────────────────────────────────────
        replay_buf.update_from_loader(
            train_loader.dataset.dataset,
            train_loader.dataset.indices,
            dataset_name=dataset_name,
            task_id=t_idx,
        )

        # ── Post-task: consolidate SI (exact from backup) ──────────────────────
        print(f"  [ANTARA] Task {t_idx} complete. Anchoring Knowledge...")
        model.memory.consolidate(task_id=t_idx, feedback_buffer=model.feedback_buffer)

        # ── Post-task: immutable anchoring (exact V23 from backup) ────────────
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
                                mask.to(p.device), old.to(p.device), p.data
                            )
                for name, b in m_tracked.named_buffers():
                    if 'running_mean' in name or 'running_var' in name:
                        prefix = name.rsplit('.', 1)[0]
                        w_name = f"{prefix}.weight"
                        is_sacred_bn = (w_name in model.memory.sacred_mask and
                                        model.memory.sacred_mask[w_name].any())
                        if not is_sacred_bn or name not in model.memory.anchor:
                            model.memory.anchor[name] = b.data.clone().detach()

        # ── Post-task: V19 mask rebuild ────────────────────────────────────────
        _backbone_ref = model.memory.models[0]
        model.memory._v19_update(model.memory, t_idx, _backbone_ref)

        total_sacred = sum(m.sum().item() for m in model.memory.param_id_to_mask.values())
        num_total    = sum(p.numel() for p in _backbone_ref.parameters() if p.requires_grad)
        model.memory.saturation_level = total_sacred / num_total
        print(f"  [SENTIENT] Locked: {total_sacred:,}/{num_total:,} ({model.memory.saturation_level:.2%})")

        # ── Post-task: reset Lookahead slow_weights (V17 from backup) ─────────
        if hasattr(model, 'slow_weights') and model.config.use_lookahead:
            model.slow_weights = {
                n: p.data.clone().detach()
                for n, p in model.model.named_parameters()
                if p.requires_grad
            }

        # ── Evaluate all seen tasks (pure Class-IL) ────────────────────────────
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

    # ── Final report ───────────────────────────────────────────────────────────
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
    parser.add_argument("--epochs",  type=int, default=None,
                        help="Epochs per task (default 20 for A100)")
    args = parser.parse_args()
    for stage in args.stages:
        for seed in args.seeds:
            run_experiment(stage_id=stage, seed=seed,
                           dataset_name=args.dataset,
                           epochs_override=args.epochs)
