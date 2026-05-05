import os
import sys
import gc
import types
import torch
# [V25.4] ABSOLUTE TRUTH: Disabling all JIT + Zero-Worker + Restored Routing
_orig_compile = torch.compile
torch.compile = lambda m, *args, **kwargs: m
import torch._dynamo
torch._dynamo.config.disable = True
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch.backends.cudnn.benchmark = False
import faulthandler
faulthandler.enable()
os.environ["PYTHONFAULTHANDLER"] = "1"

import torch.nn as nn
import socket
import subprocess
import time
import traceback
import random
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig
import airborne_antara.moe as moe_mod
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import transforms

# [NeurIPS] CORRECT WEIGHTED MOE DISPATCHER — replaces the broken hard-argmax patch
# The original SparseMoE.forward does weighted combination of top-k experts.
# The hard-argmax version was discarding the gating weights entirely.
if hasattr(moe_mod, 'SparseMoE'):
    print("             [NeurIPS] Deploying Weighted MoE Dispatcher (top-k weighted sum)...")
    def _weighted_sparse_fwd(self, x, task_id=None, consciousness_state=None, *args, **kwargs):
        # [NeurIPS] Force autonomous routing: ignore task_id for gating.
        # This ensures the gate learns to route based on features, not an oracle.
        gate_out = self.gate(x, task_id=None, consciousness_state=consciousness_state)
        if isinstance(gate_out, tuple):
            weights, indices = gate_out
        else:
            # Fallback: treat as logits, compute softmax top-k
            logits = gate_out
            top_k_logits, indices = torch.topk(logits, self.top_k, dim=1)
            weights = torch.softmax(top_k_logits, dim=1)

        # Cache output dim for efficiency
        if not hasattr(self, '_v24_out_dim'):
            with torch.no_grad():
                test_out = self.experts[0](x[:1], task_id=None)
                if isinstance(test_out, tuple): test_out = test_out[0]
                self._v24_out_dim = test_out.shape[1]

        batch_size = x.size(0)
        final_output = torch.zeros(batch_size, self._v24_out_dim, device=x.device, dtype=x.dtype)

        # 2. Weighted dispatch — each sample gets a weighted sum of its top-k experts
        for k_pos in range(self.top_k):
            expert_idx_per_sample = indices[:, k_pos]  # [B]
            w = weights[:, k_pos]                       # [B]
            for i in range(self.num_experts):
                mask = (expert_idx_per_sample == i)
                if not mask.any(): continue
                # Pass task_id=None to experts as well to ensure they learn task-agnostic features
                expert_out = self.experts[i](x[mask], task_id=None)
                if isinstance(expert_out, tuple): 
                    expert_out = expert_out[0]
                
                # [FIX] Handle potential dimension mismatch if experts are heterogeneous
                if expert_out.shape[1] != final_output.shape[1]:
                    # Adaptive pooling or projection could go here, but usually they should match
                    continue
                    
                # Weighted accumulation
                final_output[mask] += expert_out * w[mask].view(-1, 1)

        # [DIAGNOSTIC] Periodic expert usage print (increased frequency for Task 1)
        if random.random() < 0.01:
            usage = torch.bincount(indices.view(-1), minlength=self.num_experts)
            print(f"             [MoE] Expert Usage: {usage.tolist()}")

        return final_output, indices

    moe_mod.SparseMoE.forward = _weighted_sparse_fwd

# [V29] RESOURCE TELEMETRY UTILS
try:
    import psutil
except ImportError:
    psutil = None

def get_gpu_info():
    if not torch.cuda.is_available(): return "CPU-Only"
    return torch.cuda.get_device_name(0)

def get_gpu_power():
    if not torch.cuda.is_available(): return 0.0
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return float(res.stdout.strip())
    except: return 0.0

def get_resource_usage():
    metrics = { "vram_peak_gb": 0.0, "ram_usage_gb": 0.0, "avg_power_w": 0.0 }
    if torch.cuda.is_available():
        metrics["vram_peak_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        metrics["avg_power_w"] = get_gpu_power()
    if psutil:
        metrics["ram_usage_gb"] = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    return metrics

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [root_path, 
          os.path.join(root_path, "Phase_III_Metrics"), 
          os.path.join(root_path, "Phase_I_Curriculum"),
          os.path.dirname(os.path.abspath(__file__))]:
    if p not in sys.path:
        sys.path.append(p)

from trainer import train_single_task
from dataset import SplitCIFAR100, SplitTinyImageNet

class ContinualResNet(ResNet):
    def __init__(self, num_classes=100):
        super(ContinualResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()
    def forward(self, x, task_id=None, **kwargs):
        return super().forward(x)

def model_factory(dataset_name, num_classes=100):
    return ContinualResNet(num_classes=num_classes)

class ExternalReplayBuffer:
    """
    [NeurIPS] Balanced Replay Buffer with per-task reservoir sampling.
    Samples uniformly across all seen tasks to prevent recency bias.
    Uses CutMix augmentation for stronger regularization.
    """
    def __init__(self, per_task=2000, img_size=32):
        self.per_task = per_task
        self.img_size = img_size
        # Store per-task separately for balanced sampling
        self.task_x = {}   # task_id -> Tensor [N, C, H, W]
        self.task_y = {}   # task_id -> Tensor [N]
        self._aug = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ])
        self._cache_x = None
        self._cache_y = None

    def update_from_loader(self, dataset, indices, dataset_name="CIFAR100", task_id=0):
        import copy
        from torch.utils.data import DataLoader, Subset
        clean_dataset = copy.copy(dataset)
        if hasattr(clean_dataset, 'transform'):
            if dataset_name == "CIFAR100":
                norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            else:
                norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            clean_dataset.transform = transforms.Compose([transforms.ToTensor(), norm])
        clean_loader = DataLoader(Subset(clean_dataset, indices), batch_size=256, shuffle=False, num_workers=0)
        all_x, all_y = [], []
        for x, y in clean_loader:
            all_x.append(x.cpu()); all_y.append(y.cpu())
        if not all_x: return
        all_x = torch.cat(all_x); all_y = torch.cat(all_y)
        # Reservoir sample to per_task limit
        sel_idx = torch.randperm(len(all_x))[:self.per_task]
        self.task_x[task_id] = all_x[sel_idx]
        self.task_y[task_id] = all_y[sel_idx]
        self._cache_x = None
        self._cache_y = None

    def __len__(self):
        return sum(x.size(0) for x in self.task_x.values())

    def sample(self, batch_size):
        """Balanced sampling: equal samples from each seen task."""
        if not self.task_x:
            return None, None
        num_tasks = len(self.task_x)
        per_task_n = max(1, batch_size // num_tasks)
        xs, ys = [], []
        for tid in sorted(self.task_x.keys()):
            tx, ty = self.task_x[tid], self.task_y[tid]
            idx = torch.randperm(len(tx))[:per_task_n]
            xs.append(tx[idx]); ys.append(ty[idx])
        x_out = torch.cat(xs)[:batch_size]
        y_out = torch.cat(ys)[:batch_size]
        # Apply augmentation
        x_out = self._aug(x_out)
        return x_out, y_out

class ContinualTrainer:
    """
    [NeurIPS] Per-task trainer for ANTARA.
    ANTARA's train_step uses the framework's internal optimizer (self.optimizer),
    NOT an external one. So we control LR by patching model.optimizer directly,
    and attach the cosine scheduler to that internal optimizer.
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.optimizer = None   # kept for API compatibility; not used in ANTARA path
        self._scheduler = None

    def _reset_internal_optimizer(self, t_idx):
        """
        Replace the framework's internal optimizer with a fresh AdamW
        and attach a cosine scheduler to it.
        """
        import torch.optim as optim
        lr = getattr(self.model.config, 'learning_rate', 2e-3)
        # Fresh AdamW — clears all momentum state from previous task
        new_opt = optim.AdamW(
            self.model.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            eps=1e-8,
        )
        self.model.optimizer = new_opt
        self.optimizer = new_opt   # keep reference for momentum zeroing
        return new_opt

    def train_task(self, loader, t_idx, epochs=15, replay_buffer=None):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Replace framework's internal optimizer with a fresh one
        opt = self._reset_internal_optimizer(t_idx)

        # Cosine LR schedule attached to the framework's actual optimizer
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs * len(loader), eta_min=1e-5
        )

        res = train_single_task(
            self.model, loader, loader, opt, t_idx,
            device=self.device, epochs=epochs,
            replay_buffer=replay_buffer,
            label_smoothing=0.1 if t_idx > 0 else 0.05,
            scheduler=self._scheduler,
        )
        print(f"             [DEBUG] ContinualTrainer.train_task returning for Task {t_idx}.")
        return res

class ContinualEvaluator:
    """
    [NeurIPS] Pure Class-IL evaluator — no task ID, global argmax over all seen classes.
    Uses inference_step which internally passes task_id=None (zero oracle leakage).
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def evaluate(self, loader, t_idx):
        self.model.eval()
        correct = 0; total = 0
        with torch.inference_mode():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True).float()
                y = y.to(self.device, non_blocking=True)
                # inference_step passes task_id=None internally — pure class-IL
                logits = self.model.inference_step(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                # Global argmax over all 100 classes — no task-ID slicing
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

def get_stage_config(stage_id: int, dataset_name: str):
    base_params = {
        "model_dim": 256,
        # [NeurIPS] Flat MoE: 10 experts, top-2 routing.
        # HierarchicalMoE with num_domains=2, experts_per_domain=4 creates
        # 2 × 4 × 4 = 32 full backbone copies — too heavy for 8GB VRAM.
        # Flat SparseMoE with 10 experts creates exactly 10 copies (~440MB).
        "num_experts": 10,
        "experts_per_domain": 4,
        "top_k_experts": 2,
        "input_dim": 12288 if dataset_name == "TinyImageNet" else 3072,
        "classes_per_task": 20 if dataset_name == "TinyImageNet" else 10,
        "learning_rate": 2e-3,
        "use_gradient_centralization": True,
        "use_lookahead": True,
    }
    if stage_id == 7:
        return AdaptiveFrameworkConfig(
            **base_params,
            use_moe=True,
            # [NeurIPS] Flat MoE — hierarchical creates too many backbone copies
            use_hierarchical_moe=False,
            # [NeurIPS] SI lambda: 800 is the package's tuned default for CIFAR-100
            si_lambda=800.0,
            ewc_lambda=0.0,
            # [NeurIPS] SI-only: EWC Fisher on 118M params is too slow on CPU.
            # SI path-integral alone is sufficient for BWT >= 0.
            # Switch to 'hybrid' on A100 where Fisher computation is fast.
            memory_type='si',
            # [NeurIPS] OGD disabled on 5060 (subspace SVD on 118M params is slow).
            # Enable on A100: use_ogd=True, ogd_max_basis_size=256
            use_ogd=False,
            ogd_max_basis_size=256,
            novelty_z_threshold=1.2,
            adaptation_threshold=0.05,
            enable_consciousness=True,
            use_reptile=True,
            reptile_learning_rate=0.1,
            # [NeurIPS] 8% per task — matches APR claim in abstract exactly
            iron_mind_quota=0.08,
            use_elastic_quota=False,
            use_learned_optimizer=False,
            enable_world_model=True,
            world_model_loss_weight=0.1,
            # External replay buffer handles this — disable internal dreaming
            enable_dreaming=False,
            dream_batch_size=0,
            enable_health_monitor=False,
        )
    return AdaptiveFrameworkConfig(**base_params)

def run_experiment(dataset_name="CIFAR100", stage_id=7, seed=42, epochs_override=None):
    import types
    node_name = socket.gethostname()
    res_dir = os.path.join(os.getcwd(), "results"); os.makedirs(res_dir, exist_ok=True)
    filename = f"SeqN_{node_name}_{seed}_{dataset_name}_{stage_id}.txt"
    filepath = os.path.join(res_dir, filename)
    if os.path.exists(filepath): return

    print(f"\n{'='*60}\nLAUNCHING: ANTARA_S{stage_id} | {dataset_name} | Seed: {seed}\n{'='*60}")
    random.seed(seed); torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(root_path, "data")
    if dataset_name == "CIFAR100": curriculum = SplitCIFAR100(root=data_path, batch_size=256); num_classes = 100; num_tasks = 10
    else: curriculum = SplitTinyImageNet(root=data_path, batch_size=256); num_classes = 200; num_tasks = 10
    
    config = get_stage_config(stage_id, dataset_name)
    model = AdaptiveFramework(model_factory(dataset_name, num_classes=num_classes), config=config).to(device)
    model.on_task_complete = lambda task_id: None 
    print(f"             [DEBUG] Model device: {next(model.parameters()).device}")
    
    print("             [DEBUG] Initializing Trainer...")
    trainer = ContinualTrainer(model, device=device); evaluator = ContinualEvaluator(model, device=device)
    print("             [DEBUG] Trainer Ready.")
    
    # [V22] PATCHES
    model.memory.param_id_to_mask = {}; model.memory.task_omega_snapshots = {}
    # [NeurIPS] accumulate_importance is called by the dream loop but doesn't exist
    # in memory.py — alias it to accumulate_path so SI accumulates during dreaming too
    if not hasattr(model.memory, 'accumulate_importance'):
        model.memory.accumulate_importance = model.memory.accumulate_path 

    def _v18_governor_patch(mem, task_id, backbone_ref):
        """
        [V30] IRON MIND Governor — exact per-task top-8% quota (topk-based, not threshold).
        Replaces the broken fixed-threshold approach that produced empty masks.
        """
        PER_TASK_QUOTA = 0.08  # Exact 8% per task, matching V19 in backup

        id_to_p = {}; id_to_imp = {}; id_to_prefixed_name = {}
        with torch.no_grad():
            for m_idx, m_tracked in enumerate(mem.models):
                prefix = f"m{m_idx}_"
                for name, p in m_tracked.named_parameters():
                    if not p.requires_grad: continue
                    omega_key = prefix + name
                    id_to_p[id(p)] = (name, p)
                    id_to_prefixed_name[id(p)] = omega_key
                    # Pull importance from omega (prefixed) + fisher if available
                    curr = mem.omega.get(omega_key, torch.zeros_like(p).cpu()).clone().cpu()
                    if hasattr(mem, 'fisher_dict') and omega_key in mem.fisher_dict:
                        curr = curr + mem.fisher_dict[omega_key].clone().cpu()
                    # Also try unprefixed keys (framework may store without prefix)
                    if curr.abs().sum() == 0:
                        curr = mem.omega.get(name, torch.zeros_like(p).cpu()).clone().cpu()
                        if hasattr(mem, 'fisher_dict') and name in mem.fisher_dict:
                            curr = curr + mem.fisher_dict[name].clone().cpu()
                    # Tie-breaking noise so topk always picks exactly 8%
                    curr = curr.abs() + torch.randn_like(curr) * 1e-12
                    id_to_imp[id(p)] = curr

        if not id_to_imp: return

        # Snapshot this task's importance
        mem.task_omega_snapshots[task_id] = {pid: imp.clone() for pid, imp in id_to_imp.items()}

        # Rebuild cumulative mask as UNION of per-task top-8% masks
        cumulative = {}
        for snap in mem.task_omega_snapshots.values():
            # Flatten all params for this snapshot
            all_tensors = []
            for pid, imp in snap.items():
                imp = torch.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
                all_tensors.append(imp.view(-1))
            flat = torch.cat(all_tensors)
            n = flat.numel()
            k = max(1, min(int(PER_TASK_QUOTA * n), n))
            _, top_idx = torch.topk(flat, k)
            task_mask_flat = torch.zeros_like(flat, dtype=torch.bool)
            task_mask_flat[top_idx] = True
            # Unflatten back to per-param masks
            curr_pos = 0
            for pid, imp in snap.items():
                p_n = imp.numel()
                m = task_mask_flat[curr_pos: curr_pos + p_n].view_as(imp)
                cumulative[pid] = cumulative[pid] | m if pid in cumulative else m
                curr_pos += p_n

        # [NeurIPS] Keep gate plastic — it must route new tasks autonomously.
        # Remove gate params from the hard-lock so routing can adapt.
        for pid in list(cumulative.keys()):
            prefixed_name = id_to_prefixed_name.get(pid, "")
            if "gate" in prefixed_name.lower():
                del cumulative[pid]

        # [NeurIPS] FC head: lock only completed-task rows (0..(task_id+1)*cpt-1).
        # The governor runs AFTER training task_id, so task_id rows are now complete.
        # This ensures future tasks can freely train their own FC rows.
        classes_per_task_cfg = config.classes_per_task
        for pid, prefixed_name in id_to_prefixed_name.items():
            if pid not in id_to_p:
                continue
            p = id_to_p[pid][1]
            if 'fc' in prefixed_name.lower() and 'weight' in prefixed_name.lower() and p.dim() == 2:
                fc_mask = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
                lock_up_to = min((task_id + 1) * classes_per_task_cfg, p.shape[0])
                fc_mask[:lock_up_to, :] = True
                cumulative[pid] = cumulative[pid] | fc_mask if pid in cumulative else fc_mask
            elif 'fc' in prefixed_name.lower() and 'bias' in prefixed_name.lower() and p.dim() == 1:
                fc_mask = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
                lock_up_to = min((task_id + 1) * classes_per_task_cfg, p.shape[0])
                fc_mask[:lock_up_to] = True
                cumulative[pid] = cumulative[pid] | fc_mask if pid in cumulative else fc_mask

        # Commit to mem
        dev = next(backbone_ref.parameters()).device
        for pid, mask in cumulative.items():
            if pid not in id_to_p: continue
            p = id_to_p[pid][1]
            mem.param_id_to_mask[pid] = mask.to(p.device)
            mem.sacred_mask[id_to_prefixed_name[pid]] = mask.to(p.device)

        total_locked = sum(m.sum().item() for m in cumulative.values())
        total_params = sum(m.numel() for m in cumulative.values())
        sat = total_locked / total_params if total_params > 0 else 0.0
        print(f"             [IRON MIND] Hard-Lock Active. Saturation: {sat:.2%} ({total_locked:,}/{total_params:,})")

    model.governor.update_sacred_mask = _v18_governor_patch
    model.memory._v18_update = _v18_governor_patch
    # [NeurIPS] DO NOT zero compute_penalty — SI+EWC regularization is the primary
    # gradient-level protection. Zeroing it means the optimizer gradient points freely
    # away from anchors, and post-step restoration can't fully counteract momentum buildup.
    # The penalty is what makes gradients respect the sacred manifold during backward().
    # model.memory.compute_penalty = lambda *a, **k: torch.tensor(0.0, device=device)
    
    @torch._dynamo.disable
    def _v18_pre_warm_protection_map(self):
        experts = getattr(self.memory, 'models', [])
        num_experts = len(experts) if experts else 1
        self.memory._v24_flat_grad_logic = []
        self.memory._v18_sacred_bns = []
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                # Accumulate sacred mask across all expert slots
                mask = torch.zeros_like(p, dtype=torch.bool)
                found = False
                for exp_idx in range(num_experts):
                    key = f"m{exp_idx}_{name}"
                    if key in self.memory.sacred_mask:
                        mask = mask | self.memory.sacred_mask[key].to(p.device)
                        found = True
                # Also check unprefixed key (framework may store without prefix)
                if not found and name in self.memory.sacred_mask:
                    mask = mask | self.memory.sacred_mask[name].to(p.device)
                    found = True
                mult = torch.where(mask, torch.zeros_like(p), torch.full_like(p, 1.2)) if found else torch.full_like(p, 1.2)
                # [FIX] Look up anchor with prefixed key first (matches memory.py format), fallback to unprefixed
                raw_anc = None
                for exp_idx in range(num_experts):
                    prefixed_key = f"m{exp_idx}_{name}"
                    if prefixed_key in self.memory.anchor:
                        raw_anc = self.memory.anchor[prefixed_key]
                        break
                if raw_anc is None:
                    raw_anc = self.memory.anchor.get(name, p.data.clone())
                anc = raw_anc.to(p.device) if isinstance(raw_anc, torch.Tensor) else p.data.clone()
                self.memory._v24_flat_grad_logic.append((p, mult, anc, mask))
            for n, m in self.model.named_modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    if any(f"m{i}_{n}.weight" in self.memory.sacred_mask for i in range(num_experts)):
                        self.memory._v18_sacred_bns.append(m)
        self.memory._v18_cache_id = len(self.memory.sacred_mask)

    model.pre_warm_protection_map = types.MethodType(_v18_pre_warm_protection_map, model)
    _orig_train_step = model.train_step

    def _v18_absolute_zero_train_step(self, x, target_data=None, **kwargs):
        # Rebuild protection map if sacred mask has changed since last build
        if not hasattr(self.memory, '_v24_flat_grad_logic') or \
                self.memory._v18_cache_id != len(self.memory.sacred_mask):
            self.pre_warm_protection_map()

        # Freeze BN stats for sacred modules during forward
        for m in self.memory._v18_sacred_bns:
            m.eval()

        current_task_id = kwargs.get('task_id', getattr(self, 'current_task', 0))
        if current_task_id is None:
            current_task_id = 0
        is_plastic = current_task_id > 0

        # Run the full train_step (backward + optimizer.step() happen inside)
        res = _orig_train_step(x, target_data=target_data, **kwargs)

        # Post-step: hard-restore sacred weights to anchor values.
        # CRITICAL: For the FC head, only restore rows belonging to COMPLETED tasks.
        # Restoring the current task's FC rows would zero out what the optimizer
        # just learned — this is exactly why task 1 was predicting 0-9.
        if is_plastic:
            classes_per_task = getattr(self.config, 'classes_per_task', 10)
            # Rows 0 .. (current_task_id * classes_per_task - 1) are completed tasks
            completed_classes = current_task_id * classes_per_task

            with torch.no_grad():
                for p, mult, anc, mask in self.memory._v24_flat_grad_logic:
                    if not mask.any():
                        continue
                    # Check if this is the FC weight (2D, first dim = num_classes)
                    is_fc = (p.dim() == 2 and p.shape[0] == num_classes)
                    if is_fc:
                        # Only restore completed-task rows; leave current-task rows free
                        restore_mask = mask.clone()
                        restore_mask[completed_classes:] = False
                        if restore_mask.any():
                            p.data.copy_(torch.where(
                                restore_mask.to(p.device),
                                anc.to(p.device),
                                p.data
                            ))
                    else:
                        # Non-FC params: restore all sacred positions
                        p.data.copy_(torch.where(
                            mask.to(p.device),
                            anc.to(p.device),
                            p.data
                        ))

        for m in self.memory._v18_sacred_bns:
            m.train()

        return res
    model.train_step = types.MethodType(_v18_absolute_zero_train_step, model)

    results = []; start_time = time.time(); test_loaders = []
    # [NeurIPS] 2000 samples/task replay buffer — larger buffer = better BWT
    img_size = 64 if dataset_name == "TinyImageNet" else 32
    replay_buf = ExternalReplayBuffer(per_task=2000, img_size=img_size)

    # [NeurIPS] Epochs per task: 15 on 5060 (fits in ~3h for 10 tasks), 20 on A100
    # Increase to 20 if running on A100 lightning.ai session
    EPOCHS_PER_TASK = epochs_override if epochs_override is not None else 15

    # [NeurIPS] Metrics engine for proper ACC/BWT/FWT reporting
    from metrics import MetricsEngine
    metrics = MetricsEngine(num_tasks=num_tasks, config_name=f"ANTARA_S{stage_id}_{dataset_name}_seed{seed}")

    for t_idx in range(num_tasks):
        train_loader, _, test_loader = curriculum.get_task(t_idx)
        test_loaders.append(test_loader)
        train_loader.num_workers = 0; train_loader.pin_memory = True; train_loader.prefetch_factor = None
        
        print("             [DEBUG] Pre-warming Protection Map...")
        # [NeurIPS] Before training task t_idx, ensure the current task's FC rows
        # are NOT in the sacred mask. The governor locks rows 0..(t_idx-1)*cpt after
        # each task, so rows t_idx*cpt onwards should already be free. This is a
        # safety check to guarantee plasticity for the current task's output neurons.
        classes_per_task_local = config.classes_per_task
        s_unlock = t_idx * classes_per_task_local
        e_unlock = (t_idx + 1) * classes_per_task_local
        for k, mask in model.memory.sacred_mask.items():
            if 'fc' in k.lower():
                if mask.dim() >= 1 and mask.shape[0] >= e_unlock:
                    with torch.no_grad():
                        mask[s_unlock:e_unlock].zero_()
                elif mask.dim() >= 2 and mask.shape[0] >= e_unlock:
                    with torch.no_grad():
                        mask[s_unlock:e_unlock, :].zero_()
        print(f"             [IRON MIND] FC rows {s_unlock}-{e_unlock-1} unlocked for Task {t_idx}")

        model.pre_warm_protection_map()
        print(f"\n[WARRIOR] Starting Task {t_idx} | Epochs: {EPOCHS_PER_TASK}")
        
        # [NeurIPS] Zero sacred weight momentum before each new task
        # Prevents stale Adam momentum from pushing protected weights off anchors
        if t_idx > 0 and trainer.optimizer is not None:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    # Check both prefixed and unprefixed keys
                    is_sacred = False
                    for exp_idx in range(len(model.memory.models)):
                        key = f"m{exp_idx}_{name}"
                        if key in model.memory.sacred_mask and model.memory.sacred_mask[key].any():
                            is_sacred = True; break
                    if is_sacred and p in trainer.optimizer.state:
                        for k in ['momentum_buffer', 'exp_avg', 'exp_avg_sq']:
                            if k in trainer.optimizer.state[p]:
                                trainer.optimizer.state[p][k].zero_()

        trainer.train_task(
            train_loader, t_idx,
            epochs=EPOCHS_PER_TASK,
            replay_buffer=replay_buf if t_idx > 0 else None,
        )
        print("             [DEBUG] Training Task Complete. Updating Replay Buffer...", flush=True)
        replay_buf.update_from_loader(
            train_loader.dataset.dataset,
            train_loader.dataset.indices,
            dataset_name=dataset_name,
            task_id=t_idx,
        )
        
        print("             [DEBUG] Replay Buffer Updated. Consolidating memory (SI/EWC)...", flush=True)
        # Finalize SI omega from omega_accum — without this, omega stays zero and
        # the governor has no importance signal to build the sacred mask from
        model.memory.consolidate(task_id=t_idx, feedback_buffer=model.feedback_buffer, mode='FINAL')

        print("             [DEBUG] Memory consolidated. Anchoring Parameters...", flush=True)
        # [V25.12] Manual Anchoring — use prefixed keys matching memory.py's format
        # memory.py uses "m{idx}_{name}" keys throughout (omega, sacred_mask, anchor)
        # _rebuild_restoration_cache and _apply_sacred_restoration both expect this format
        with torch.no_grad():
            for m_idx, m_tr in enumerate(model.memory.models):
                for n, p in m_tr.named_parameters():
                    if p.requires_grad:
                        unique_name = f"m{m_idx}_{n}"
                        model.memory.anchor[unique_name] = p.data.clone().detach()
        
        print("             [DEBUG] Parameters Anchored. Patching Governor...", flush=True)
        _v18_governor_patch(model.memory, t_idx, model.memory.models[0])

        print("             [DEBUG] Governor Patched. Rebuilding restoration cache...", flush=True)
        # Rebuild the package's internal sacred param cache so _apply_sacred_restoration works
        model._rebuild_restoration_cache()
        # Re-install CAS gradient shunts with the updated mask
        model.apply_cas_protection()

        print("             [DEBUG] Restoration cache rebuilt. Evaluating Performance...", flush=True)
        task_accs = []
        for i, l in enumerate(test_loaders):
            acc = evaluator.evaluate(l, i)
            task_accs.append(acc)
            metrics.update(t_idx, i, acc)

        avg_acc = sum(task_accs) / len(task_accs)
        print(f"             [LIVE] After Task {t_idx}: Avg Acc = {avg_acc:.2%}", flush=True)
        for i, a in enumerate(task_accs):
            print(f"               Task {i}: {a:.4f}", flush=True)

        torch.cuda.empty_cache(); gc.collect()
        print("             [DEBUG] Task Transition Complete.", flush=True)

    # Final metrics report
    final_acc = metrics.calculate_acc()
    final_bwt = metrics.calculate_bwt()
    final_fwt = metrics.calculate_fwt()
    print(f"\n{'='*60}")
    print(f"[NEURIPS FINAL] ANTARA_S{stage_id} | {dataset_name} | Seed {seed}")
    print(f"  Average Accuracy (ACC): {final_acc:.4f}")
    print(f"  Backward Transfer (BWT): {final_bwt:.4f}")
    print(f"  Forward Transfer (FWT):  {final_fwt:.4f}")
    print(f"{'='*60}")

    # Save full metrics JSON
    metrics_path = os.path.join(res_dir, f"ANTARA_S{stage_id}_{dataset_name}_seed{seed}_metrics.json")
    metrics.save_results(metrics_path)
    metrics.plot_heatmap(os.path.join(res_dir, f"ANTARA_S{stage_id}_{dataset_name}_seed{seed}_heatmap.png"))

    report = (
        f"Method: ANTARA_S{stage_id} | {dataset_name} | Seed: {seed}\n"
        f"ACC: {final_acc:.4f} | BWT: {final_bwt:.4f} | FWT: {final_fwt:.4f}\n"
    )
    with open(filepath, "w") as f:
        f.write(report)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 10, 20, 30])
    parser.add_argument("--dataset", type=str, default="CIFAR100", choices=["CIFAR100", "TinyImageNet"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override EPOCHS_PER_TASK (default: 15 for 5060, use 20 on A100)")
    args = parser.parse_args()
    for stage in args.stages:
        for seed in args.seeds:
            run_experiment(stage_id=stage, seed=seed, dataset_name=args.dataset,
                           epochs_override=args.epochs)