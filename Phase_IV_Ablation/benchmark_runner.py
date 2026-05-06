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
- Replay buffer: 500 samples/task (stronger signal for Class-IL)
- Fisher importance: 256 samples (4x stronger than baseline)
- 20 epochs per task
"""
import os
import sys
import gc
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

        # Track expert usage (needed for expert dedication / freezing)
        with torch.no_grad():
            flat_indices = indices.view(-1)
            self.expert_usage.index_add_(
                0, flat_indices,
                torch.ones_like(flat_indices, dtype=self.expert_usage.dtype)
            )

        # Block routing to frozen experts during training (expert dedication)
        # Frozen experts are dedicated to previous tasks — new tasks must use free experts.
        # We must clone weights before modifying to avoid in-place autograd graph corruption.
        if self.training:
            frozen_mask = torch.zeros(self.num_experts, dtype=torch.bool, device=x.device)
            for i, expert in enumerate(self.experts):
                if all(not p.requires_grad for p in expert.parameters()):
                    frozen_mask[i] = True
            if frozen_mask.any() and not frozen_mask.all():
                # Build a per-sample, per-k mask of frozen selections
                # indices shape: (B, top_k)
                is_frozen_selection = frozen_mask[indices]  # (B, top_k) bool
                # Zero out weights for frozen expert selections, re-normalize
                # Use out-of-place ops to preserve autograd graph
                free_weights = weights * (~is_frozen_selection).float()
                weight_sum = free_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
                weights = free_weights / weight_sum

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

# ── Latent Consistency Loss helpers ───────────────────────────────────────────
def _get_resnet_backbone(framework_model):
    """
    Navigate the AdaptiveFramework → SparseMoE → Expert → ContinualResNet path.
    When use_moe=True, framework_model.model is SparseMoE, and the backbone
    lives at model.model.experts[0].model (the first expert's base model).
    """
    m = framework_model.model
    # SparseMoE path: experts[0].model is the ContinualResNet
    if hasattr(m, 'experts') and len(m.experts) > 0:
        expert0 = m.experts[0]
        if hasattr(expert0, 'model'):
            return expert0.model
        # Some versions wrap differently
        for child in expert0.children():
            if hasattr(child, 'conv1'):
                return child
    # Fallback: direct model
    if hasattr(m, 'conv1'):
        return m
    raise RuntimeError(f"Cannot find ContinualResNet backbone in {type(m).__name__}")

def _get_all_expert_backbones(framework_model):
    """
    Return all ContinualResNet backbones from all MoE experts.
    Each expert is independent — we must anchor ALL of them.
    """
    m = framework_model.model
    if hasattr(m, 'experts'):
        backbones = []
        for expert in m.experts:
            if hasattr(expert, 'model') and hasattr(expert.model, 'conv1'):
                backbones.append(expert.model)
        if backbones:
            return backbones
    # Fallback: single backbone
    if hasattr(m, 'conv1'):
        return [m]
    return [_get_resnet_backbone(framework_model)]

def _extract_features(backbone, x):
    """
    Extract penultimate-layer features (avgpool output, before FC).
    This is z(x) in the stability bound: δ = sup‖z(x) − z₀(x)‖.
    Works with ContinualResNet (standard ResNet-18 layout).
    """
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    x = backbone.avgpool(x)
    return torch.flatten(x, 1)   # (B, 512)

def _extract_early_features(backbone, x):
    """
    Extract early-layer features only (after layer2).
    Early layers (conv1, layer1, layer2) are universal feature detectors
    that should be preserved across tasks. Later layers (layer3, layer4)
    need plasticity to adapt to new task distributions.
    Anchoring only early layers balances stability vs plasticity.
    """
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    return x.mean(dim=[2, 3])  # Global average pool → (B, 128)

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
            # SI only — we compute Fisher ourselves (fast, vectorized)
            si_lambda=1.0,
            ewc_lambda=0.0,
            memory_type='si',
            use_ogd=False,
            use_reptile=False,
            reptile_learning_rate=0.1,
            iron_mind_quota=0.08,
            use_elastic_quota=False,
            use_learned_optimizer=False,
            enable_consciousness=False,
            enable_world_model=False,
            world_model_loss_weight=0.0,
            enable_dreaming=False,
            dream_batch_size=0,
            enable_health_monitor=False,
            feedback_buffer_size=64,
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

    # Per-task quota registry — stored so each task's mask is always evaluated
    # with the quota it was originally assigned, not the current task's quota.
    _task_quota_registry = {}

    def _v19_update(mem, task_id, backbone_ref):
        """
        Top-K importance mask per task, union across tasks.
        Task 0 gets 30% quota (needs to protect enough backbone for class-IL).
        Tasks 1-9 get 8% each (standard APR quota).
        FC rows and gate rows hard-locked for completed tasks.
        Each task's quota is stored in _task_quota_registry so the union
        always re-evaluates each snapshot with its ORIGINAL quota, not the
        current task's quota (fixes the saturation collapse bug).
        """
        # Task 0 needs a larger quota — it's the only task that trains the full backbone
        # from scratch. 8% is not enough to protect task 0's features in class-IL.
        # 30% ensures the critical feature detectors are locked before task 1 trains.
        PER_TASK_QUOTA = 0.30 if task_id == 0 else 0.08
        _task_quota_registry[task_id] = PER_TASK_QUOTA

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
    # Build a direct pid->param lookup for O(1) restoration
    _pid_to_param = {}
    for tracked in model.memory.models:
        for p in tracked.parameters():
            if p.requires_grad:
                _pid_to_param[id(p)] = p

    def _make_grad_hook(p_id, mem):
        """Zero gradient for sacred positions."""
        def hook(grad):
            m = mem.param_id_to_mask.get(p_id)
            if m is not None and m.any():
                return grad * (~m.to(grad.device))
            return grad
        return hook

    hook_handles = []
    for p in _pid_to_param.values():
        h = p.register_hook(_make_grad_hook(id(p), model.memory))
        hook_handles.append(h)
    print(f"  [IRON MIND] {len(hook_handles)} absolute gradient locks installed.")

    # Sacred anchor store — pid -> (param_ref, anchor_tensor, mask)
    # Built after each task's V19 update for O(1) restoration
    sacred_restore_list = []  # list of (param, mask, anchor) tuples

    def _rebuild_restore_list():
        """Rebuild the fast restoration list after mask changes."""
        sacred_restore_list.clear()
        for pid, p in _pid_to_param.items():
            mask = model.memory.param_id_to_mask.get(pid)
            if mask is not None and mask.any():
                anchor = p.data.clone().detach()
                sacred_restore_list.append((p, mask, anchor))

    def _hard_restore_sacred():
        """Snap all sacred params back to anchor. O(n_sacred) not O(n_params²)."""
        with torch.no_grad():
            for p, mask, anchor in sacred_restore_list:
                p.data[mask] = anchor[mask]

    # Re-wire the framework's _apply_sacred_restoration to use our sacred_restore_list.
    # The framework calls this in 3 places inside train_step (post-optimizer, post-dream,
    # post-replay). We must hook all 3 sites — wrapping train_step only catches the final
    # return, missing the intermediate restorations.
    def _framework_sacred_restore(self_model=None):
        _hard_restore_sacred()
    model._apply_sacred_restoration = _framework_sacred_restore

    # Patch gradient centralization AND gradient noise injection to re-zero
    # sacred gradients after each operation.
    # Both run AFTER backward() hooks have zeroed sacred grads, and BEFORE
    # optimizer.step() — so they can re-introduce gradients at sacred positions.
    _orig_gc = model._apply_gradient_centralization
    def _gc_with_sacred_zero(self_model=None):
        _orig_gc()
        # Re-zero gradients at sacred positions after GC
        with torch.no_grad():
            for pid, p in _pid_to_param.items():
                if p.grad is None: continue
                mask = model.memory.param_id_to_mask.get(pid)
                if mask is not None and mask.any():
                    p.grad.data[mask] = 0.0
    model._apply_gradient_centralization = _gc_with_sacred_zero

    # Patch the gradient noise injection (lines 177-182 in core.py train_step).
    # It adds randn noise to ALL gradients for first 100 steps of each task,
    # which re-introduces noise at sacred positions after hooks zeroed them.
    # We neutralize it by zeroing sacred positions after the noise is added.
    # We do this by patching the forward method to install a post-backward hook
    # that fires AFTER the noise injection but BEFORE unscale_.
    # The cleanest approach: override _steps_since_task_start to always be >= 100
    # so the noise injection condition (steps_since_start < 100) is never true.
    model._steps_since_task_start = 100  # disable noise injection permanently

    # ── Training setup ─────────────────────────────────────────────────────────
    EPOCHS     = epochs_override if epochs_override is not None else 20

    # ── Latent Consistency Loss state ──────────────────────────────────────────
    # Component (iii) from the ANTARA abstract: anchors internal representations
    # against distributional drift. Directly minimises δ = sup‖z(x)−z₀(x)‖
    # from the stability bound. α controls the consistency coefficient.
    # We anchor only early layers (conv1→layer2) to preserve universal feature
    # detectors while allowing later layers to adapt for new tasks.
    _frozen_experts = []   # list of frozen ContinualResNet per expert
    _LC_ALPHA        = 1.0    # α — conservative to preserve plasticity
    _LC_BATCH        = 64     # replay samples per LCL backward pass

    # ── Expert dedication state ────────────────────────────────────────────────
    # After each task, freeze the top-2 most-used experts so they can't be
    # overwritten by subsequent tasks. This is the PackNet-style hard protection
    # applied at the expert level — no soft regularization needed.
    # With 10 experts and 10 tasks, each task gets ~1 dedicated expert.
    _task_expert_map = {}   # task_id -> list of frozen expert indices
    _EXPERTS_PER_TASK = 2   # freeze 2 experts per task — matches top_k=2
    # With 10 experts and top_k=2: tasks 0-4 get 2 dedicated experts each (5 tasks)
    # Tasks 5-9 share the remaining experts with replay-based protection
    _moe = model.model if hasattr(model.model, 'experts') else None
    print(f"  [MoE] model.model type: {type(model.model).__name__}, _moe set: {_moe is not None}")

    # Gate column protection: when an expert is frozen, also freeze the gate's
    # output column for that expert so routing is preserved at inference time.
    # Without this, the gate forgets to route old-task inputs to frozen experts.
    _frozen_expert_set = set()

    if _moe is not None and hasattr(_moe, 'gate') and hasattr(_moe.gate, 'gate'):
        _gate_linear = _moe.gate.gate  # nn.Linear(input_dim, num_experts)

        def _gate_col_hook(grad):
            if not _frozen_expert_set:
                return grad
            g = grad.clone()
            for idx in _frozen_expert_set:
                if idx < g.shape[0]:
                    g[idx] = 0.0  # gate.weight: (num_experts, input_dim)
            return g

        _gate_linear.weight.register_hook(_gate_col_hook)
        if _gate_linear.bias is not None:
            def _gate_bias_hook(grad):
                if not _frozen_expert_set:
                    return grad
                g = grad.clone()
                for idx in _frozen_expert_set:
                    if idx < g.shape[0]:
                        g[idx] = 0.0
                return g
            _gate_linear.bias.register_hook(_gate_bias_hook)
        print(f"  [MoE] Gate column protection hooks installed.")

    replay_buf = ExternalReplayBuffer(
        per_task=500,
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
        print(f"  [IRON MIND] FC rows {s}-{e-1} unlocked for Task {t_idx}")

        print(f"\n[WARRIOR] Task {t_idx} | Epochs: {EPOCHS}")

        # Disable gradient noise injection for this task
        model._steps_since_task_start = 100

        # Reset expert usage counters so we track which experts this task uses
        if _moe is not None and hasattr(_moe, 'reset_usage'):
            _moe.reset_usage()
            print(f"  [MoE] Expert usage counters reset for Task {t_idx}")
        elif _moe is not None:
            # Fallback: reset manually
            _moe.expert_usage.zero_()
            print(f"  [MoE] Expert usage reset (manual) for Task {t_idx}")

        # Fresh AdamW — only trainable params (excludes frozen experts)
        lr = config.learning_rate if t_idx == 0 else config.learning_rate * 0.5
        trainable_params = [p for p in model.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=1e-4, eps=1e-8
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

        # ── Latent Consistency Loss injection ─────────────────────────────────
        # Wrap train_step to add LCL backward after each optimizer step.
        # LCL directly minimises δ = sup‖z(x)−z₀(x)‖ from the stability bound
        # by penalising feature drift on replay samples.
        # This constrains the FREE (non-sacred) weights — the confirmed source
        # of forgetting (sacred drift = 0.000000, confirmed by diagnostic).
        import types as _types
        _orig_train_step = model.train_step

        def _lcl_train_step(self_fw, x, target_data=None, **kwargs):
            res = _orig_train_step(x, target_data=target_data, **kwargs)
            # Only apply LCL when frozen anchors exist and replay buffer has data
            if _frozen_experts and replay_buf and len(replay_buf) > 0:
                rx, ry = replay_buf.sample(_LC_BATCH)
                if rx is not None:
                    rx = rx.to(device).float()
                    live_experts = _get_all_expert_backbones(model)
                    total_lc = torch.tensor(0.0, device=device)
                    for exp_idx, (frozen_exp, live_exp) in enumerate(
                            zip(_frozen_experts, live_experts)):
                        # Frozen early-layer features — no grad
                        with torch.no_grad():
                            z_frozen = _extract_early_features(frozen_exp, rx)
                        # Current early-layer features — with grad
                        live_exp.train()
                        z_current = _extract_early_features(live_exp, rx)
                        # LCL on early layers only — preserves universal detectors
                        # while allowing later layers to adapt for new tasks
                        total_lc = total_lc + (
                            1.0 - F.cosine_similarity(z_current, z_frozen, dim=1)
                        ).mean()
                    lc_loss = _LC_ALPHA * total_lc / max(1, len(_frozen_experts))
                    lc_loss.backward()
                    # Re-zero sacred gradients after LCL backward
                    with torch.no_grad():
                        for pid, p in _pid_to_param.items():
                            if p.grad is None: continue
                            mask = model.memory.param_id_to_mask.get(pid)
                            if mask is not None and mask.any():
                                p.grad.data[mask] = 0.0
                    # Step optimizer with LCL gradients
                    model.optimizer.step()
                    model.optimizer.zero_grad()
            return res

        model.train_step = _types.MethodType(_lcl_train_step, model)

        # ── Drift diagnostic: snapshot sacred weights before training ──────────
        if t_idx > 0 and sacred_restore_list:
            _pre_train_sacred = {id(p): p.data[mask].clone() for p, mask, _ in sacred_restore_list}
        else:
            _pre_train_sacred = {}

        train_single_task(
            model, train_loader, train_loader, optimizer, t_idx,
            device=device, epochs=EPOCHS,
            replay_buffer=replay_buf if t_idx > 0 else None,
            label_smoothing=0.05 if t_idx == 0 else 0.1,
            scheduler=scheduler,
        )

        # Restore original train_step
        model.train_step = _orig_train_step

        # ── Drift diagnostic: measure how much sacred weights moved ───────────
        if _pre_train_sacred and sacred_restore_list:
            total_drift = 0.0; total_els = 0
            for p, mask, _ in sacred_restore_list:
                pid = id(p)
                if pid in _pre_train_sacred:
                    drift = (p.data[mask] - _pre_train_sacred[pid]).abs().mean().item()
                    total_drift += drift; total_els += 1
            avg_drift = total_drift / total_els if total_els > 0 else 0.0
            print(f"  [DRIFT] Task {t_idx}: avg sacred weight drift = {avg_drift:.6f} "
                  f"({'DRIFTING!' if avg_drift > 1e-6 else 'HELD'})")

        # Update replay buffer
        replay_buf.update_from_loader(
            train_loader.dataset.dataset,
            train_loader.dataset.indices,
            dataset_name=dataset_name,
            task_id=t_idx,
        )

        # ── Fast Fisher + SI consolidation (bypasses slow package consolidate) ──
        # The package's consolidate with hybrid mode runs 2500+ backward passes.
        # We compute Fisher ourselves: one vectorized backward pass over 64 samples.
        print(f"  [ANTARA] Task {t_idx} complete. Computing importance (fast Fisher+SI)...")

        # Step 1: SI consolidation only (fast — no backward pass needed)
        model.memory.consolidate(task_id=t_idx, feedback_buffer=model.feedback_buffer)

        # Step 2: Fast diagonal Fisher on replay buffer (256 samples, one batch)
        if t_idx == 0 or len(replay_buf) > 0:
            _backbone = model.memory.models[0]
            _backbone.eval()
            # Sample up to 256 items from replay buffer (or current task loader)
            if t_idx == 0:
                # Use a small subset of the current task's training data
                _fisher_x, _fisher_y = [], []
                for _bx, _by in train_loader:
                    _fisher_x.append(_bx); _fisher_y.append(_by)
                    if sum(t.size(0) for t in _fisher_x) >= 256: break
                _fisher_x = torch.cat(_fisher_x)[:256].to(device).float()
                _fisher_y = torch.cat(_fisher_y)[:256].to(device)
            else:
                _fx, _fy = replay_buf.sample(256)
                _fisher_x = _fx.to(device).float() if _fx is not None else None
                _fisher_y = _fy.to(device) if _fy is not None else None

            if _fisher_x is not None:
                # Single forward+backward to get gradient²  (diagonal Fisher)
                _backbone.zero_grad()
                with torch.enable_grad():
                    _out = _backbone(_fisher_x)
                    if isinstance(_out, tuple): _out = _out[0]
                    _loss = F.cross_entropy(_out, _fisher_y)
                    _loss.backward()

                with torch.no_grad():
                    for name, p in _backbone.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            # Fisher diagonal = grad²
                            fisher_diag = p.grad.data.pow(2)
                            # Add to omega (SI already accumulated there)
                            existing = model.memory.omega.get(name, torch.zeros_like(p))
                            model.memory.omega[name] = existing + fisher_diag * 400.0
                _backbone.zero_grad()
                _backbone.train()
                print(f"  [FISHER] Fast diagonal Fisher computed on {_fisher_x.size(0)} samples.")
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

        # Rebuild the fast restoration list with current param values as anchors.
        # IMMUTABLE RULE: once a position is in sacred_restore_list, its anchor
        # value NEVER changes — it stays at the value from when it was first locked.
        # New positions added this task get anchored at their current (just-trained) values.
        # Capture old anchors and masks BEFORE _rebuild_restore_list() clears the list.
        old_anchor_map = {id(p): (mask.clone(), anc.clone()) for p, mask, anc in sacred_restore_list}
        _rebuild_restore_list()
        # For positions that were already sacred, restore their OLD anchor values.
        # For newly-sacred positions, keep the NEW anchor values (current trained values).
        for i, (p, new_mask, new_anchor) in enumerate(sacred_restore_list):
            pid = id(p)
            if pid in old_anchor_map:
                old_mask, old_anchor = old_anchor_map[pid]
                # Merge: old positions keep old anchors, new positions keep new anchors
                merged_anchor = new_anchor.clone()
                merged_anchor[old_mask] = old_anchor[old_mask]
                sacred_restore_list[i] = (p, new_mask, merged_anchor)

        total_sacred = sum(m.sum().item() for m in model.memory.param_id_to_mask.values())
        num_total    = sum(p.numel() for p in _backbone_ref.parameters() if p.requires_grad)
        model.memory.saturation_level = total_sacred / num_total
        print(f"  [SENTIENT] Locked: {total_sacred:,}/{num_total:,} ({model.memory.saturation_level:.2%})")

        # ── Update frozen expert anchors after each task ──────────────────────
        # Capture ALL expert backbones as frozen anchors.
        # After Task 0: anchor = Task 0 features for all experts.
        # After Task N: anchor = best-so-far features (Task 0 anchors preserved
        # for expert positions that were sacred; new positions get Task N values).
        # This ensures LCL anchors ALL seen tasks, not just Task 0.
        _live_experts = _get_all_expert_backbones(model)
        if not _frozen_experts:
            # First time: create frozen copies for all experts
            for live_exp in _live_experts:
                frozen = model_factory(dataset_name, num_classes).to(device)
                frozen.load_state_dict(live_exp.state_dict(), strict=False)
                frozen.eval()
                for _p in frozen.parameters():
                    _p.requires_grad = False
                _frozen_experts.append(frozen)
            print(f"  [LCL] {len(_frozen_experts)} expert anchors frozen after Task {t_idx}.")
        else:
            # Update: for each expert, update only the NON-sacred positions
            # (sacred positions keep their original Task 0 anchor values)
            for exp_idx, (frozen_exp, live_exp) in enumerate(
                    zip(_frozen_experts, _live_experts)):
                with torch.no_grad():
                    for (fname, fp), (lname, lp) in zip(
                            frozen_exp.named_parameters(),
                            live_exp.named_parameters()):
                        # Only update non-sacred positions in the frozen anchor
                        mask = model.memory.sacred_mask.get(fname)
                        if mask is None:
                            fp.data.copy_(lp.data)
                        else:
                            # Keep sacred positions frozen, update free positions
                            fp.data[~mask] = lp.data[~mask]
            print(f"  [LCL] Expert anchors updated after Task {t_idx} "
                  f"(free positions refreshed, sacred positions held).")

        # ── Expert dedication: freeze top-used experts after each task ──────────
        if _moe is not None:
            # Get already frozen experts
            if hasattr(_moe, 'get_frozen_experts'):
                already_frozen = _moe.get_frozen_experts()
            else:
                already_frozen = [i for i, exp in enumerate(_moe.experts)
                                  if all(not p.requires_grad for p in exp.parameters())]

            # Only freeze if we have enough free experts left for future tasks
            # Keep at least 1 free expert available
            free_count = len(_moe.experts) - len(already_frozen)
            experts_to_freeze = min(_EXPERTS_PER_TASK, max(0, free_count - 1))

            if experts_to_freeze > 0:
                if hasattr(_moe, 'get_top_experts'):
                    top_experts = _moe.get_top_experts(experts_to_freeze + len(already_frozen))
                else:
                    _, top_idx = torch.topk(_moe.expert_usage, experts_to_freeze + len(already_frozen))
                    top_experts = top_idx.tolist()

                new_to_freeze = [e for e in top_experts if e not in already_frozen][:experts_to_freeze]
            else:
                new_to_freeze = []
            if new_to_freeze:
                if hasattr(_moe, 'freeze_experts'):
                    _moe.freeze_experts(new_to_freeze)
                else:
                    for idx in new_to_freeze:
                        for p in _moe.experts[idx].parameters():
                            p.requires_grad = False
                _task_expert_map[t_idx] = new_to_freeze
                _frozen_expert_set.update(new_to_freeze)  # protect gate columns too
                usage_vals = [int(_moe.expert_usage[i].item()) for i in new_to_freeze]
                print(f"  [MoE] Task {t_idx} experts frozen: {new_to_freeze} (usage: {usage_vals})")

            if hasattr(_moe, 'get_frozen_experts'):
                frozen_all = _moe.get_frozen_experts()
            else:
                frozen_all = [i for i, exp in enumerate(_moe.experts)
                              if all(not p.requires_grad for p in exp.parameters())]
            print(f"  [MoE] Total frozen: {frozen_all} | Free: {len(_moe.experts)-len(frozen_all)}/{len(_moe.experts)}")

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
