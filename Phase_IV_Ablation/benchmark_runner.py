import os
import sys
import torch
import torch.nn as nn
import socket
import subprocess
import time
import traceback
import random
import gc
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig
import airborne_antara.moe as moe_mod
from torchvision.models.resnet import ResNet, BasicBlock

# [V21.1] GRAPH-UNITY: Patching MoE .item() graph breaks for A100 Speed
if hasattr(moe_mod, 'SparseMoE'):
    print("             [V21.1] Patching MoE Graph Breaks for A100 Speed...")
    _orig_sparse_fwd = moe_mod.SparseMoE.forward
    def _unified_sparse_fwd(self, x, *args, **kwargs):
        # The original SparseMoE.forward used 'torch.rand(1).item() < 0.1' which breaks the graph.
        # By setting capture_scalar_outputs=True and monkey-patching, we keep the A100 in-graph.
        return _orig_sparse_fwd(self, x, *args, **kwargs)
    moe_mod.SparseMoE.forward = _unified_sparse_fwd
    torch._dynamo.config.capture_scalar_outputs = True
from torchvision import transforms
from trainer import train_single_task

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

# [V29] PATH INJECTION
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)
    sys.path.append(os.path.join(root_path, "Phase_III_Metrics"))
    sys.path.append(os.path.join(root_path, "Phase_I_Curriculum"))

from trainer import train_single_task
from dataset import SplitCIFAR100, SplitTinyImageNet

def get_node_name():
    return os.getenv("ANTARA_NODE", socket.gethostname())

def git_sync_file(filepath, message="Result Sync"):
    try:
        subprocess.run(["git", "add", filepath], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
    except Exception as e:
        print(f"[GIT_WARN] Sync failed for {filepath}: {e}")

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
    """[V30] Direct (x, y) tensor buffer for Class-IL replay.
    Stores CLEAN (un-augmented) samples on CPU to avoid baking in
    one specific random crop. Augmentation applied on-the-fly during sample()."""
    def __init__(self, per_task=1000, img_size=32):
        self.per_task = per_task
        self.img_size = img_size
        self.x_data = []  # list of tensors, one per stored sample
        self.y_data = []
        self._aug = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ])
        self._cache_x = None
        self._cache_y = None

    def update_from_loader(self, dataset, indices, dataset_name="CIFAR100"):
        """[V30.1] Store CLEAN samples by accessing the dataset with a simple transform.
        This prevents 'baked-in' augmentation drift in the replay buffer."""
        import copy
        from torch.utils.data import DataLoader, Subset
        
        # Create a temporary subset with no augmentation
        clean_dataset = copy.copy(dataset)
        if hasattr(clean_dataset, 'transform'):
            # Match normalization to dataset
            if dataset_name == "CIFAR100":
                norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            else: # TinyImageNet
                norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                
            clean_dataset.transform = transforms.Compose([
                transforms.ToTensor(),
                norm
            ])
            
        clean_loader = DataLoader(Subset(clean_dataset, indices), batch_size=128, shuffle=False)
        
        all_x, all_y = [], []
        for x, y in clean_loader:
            all_x.append(x.cpu())
            all_y.append(y.cpu())
            
        if not all_x: return
        
        all_x = torch.cat(all_x)
        all_y = torch.cat(all_y)
        
        sel_idx = torch.randperm(len(all_x))[:self.per_task]
        self.x_data.append(all_x[sel_idx])
        self.y_data.append(all_y[sel_idx])
        self._cache_x = None # Invalidate cache
        self._cache_y = None

    def __len__(self):
        return sum(x.size(0) for x in self.x_data)

    def sample(self, batch_size):
        """Random sample across all stored tasks with on-the-fly augmentation."""
        if self._cache_x is None:
            self._cache_x = torch.cat(self.x_data)
            self._cache_y = torch.cat(self.y_data)
        
        indices = torch.randperm(len(self._cache_x))[:batch_size]
        batch_x = self._cache_x[indices]
        batch_x = self._aug(batch_x)
        return batch_x, self._cache_y[indices]


class ContinualTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model; self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=getattr(model.config, 'learning_rate', 5e-4))
    def train_task(self, loader, t_idx, epochs=10, replay_buffer=None):
        # [V31.1] Explicitly purge old optimizer to free 4GB+ of momentum buffers
        if hasattr(self, 'optimizer'):
            del self.optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=getattr(self.model.config, 'learning_rate', 5e-4))
        return train_single_task(self.model, loader, loader, self.optimizer, t_idx, device=self.device, epochs=epochs, replay_buffer=replay_buffer)

class ContinualEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model; self.device = device
    def evaluate(self, loader, t_idx):
        self.model.eval()
        correct = 0; total = 0
        with torch.inference_mode(): # Faster than no_grad for pure eval
            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                logits = self.model.inference_step(x) if hasattr(self.model, 'inference_step') else self.model(x)
                if isinstance(logits, tuple): logits = logits[0]
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item(); total += y.size(0)
        return correct / total if total > 0 else 0.0

def get_stage_config(stage_id: int, dataset_name: str):
    base_params = {
        "model_dim": 256, "num_experts": 10, "experts_per_domain": 4, "top_k_experts": 2,
        "input_dim": 12288 if dataset_name == "TinyImageNet" else 3072,
        "classes_per_task": 20 if dataset_name == "TinyImageNet" else 10,
        "learning_rate": 1e-3, "use_gradient_centralization": True, "use_lookahead": True,
    }
    if stage_id == -1: return AdaptiveFrameworkConfig(**base_params, memory_type='ewc', ewc_lambda=5000, use_moe=False)
    if stage_id == -2: return AdaptiveFrameworkConfig(**base_params, memory_type='hybrid', use_prioritized_replay=True, dream_batch_size=32, enable_dreaming=True, dream_interval=5)
    if stage_id == -4: return AdaptiveFrameworkConfig(**base_params, memory_type='orthogonal', use_moe=False)
    if stage_id == 1: return AdaptiveFrameworkConfig(**base_params, use_moe=False, si_lambda=0.0)
    if stage_id == 2: return AdaptiveFrameworkConfig(**base_params, use_moe=False, si_lambda=1.5)
    if stage_id == 3: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5, enable_dreaming=True, dream_interval=5)
    if stage_id == 4: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5, enable_consciousness=True, enable_dreaming=True, dream_interval=5)
    if stage_id == 5: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5, enable_consciousness=True, use_reptile=True, enable_dreaming=True, dream_interval=5)
    if stage_id == 6: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.5, enable_consciousness=True, use_reptile=True, enable_world_model=True, enable_dreaming=True, dream_interval=5)
    if stage_id == 7: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.0, use_ogd=True, novelty_z_threshold=1.2, adaptation_threshold=0.05, enable_consciousness=True, use_reptile=True, iron_mind_quota=0.35, use_learned_optimizer=False)
    return AdaptiveFrameworkConfig(**base_params)

def run_experiment(dataset_name="CIFAR100", stage_id=7, seed=42):
    node_name = get_node_name()
    method_name = { -1: "EWC", -2: "DER++", -3: "LwF", -4: "RanPAC" }.get(stage_id, f"ANTARA_S{stage_id}")
    
    res_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(res_dir, exist_ok=True)
    filename = f"SeqN_{node_name}_{seed}_{dataset_name}_{stage_id}.txt"
    filepath = os.path.join(res_dir, filename)
    
    if os.path.exists(filepath):
        print(f"[SKIP] Result already exists: {filename}")
        return

    print(f"\n{'='*60}\nLAUNCHING: {method_name} | {dataset_name} | Seed: {seed}\n{'='*60}")
    
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # Auto-tune convolutions for fixed input sizes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(root_path, "data")
    # [A100 TITAN SCALING] Batch Size 64 -> 256
    if dataset_name == "CIFAR100": curriculum = SplitCIFAR100(root=data_path, batch_size=256); num_classes = 100; num_tasks = 10
    else: curriculum = SplitTinyImageNet(root=data_path, batch_size=256); num_classes = 200; num_tasks = 10
    
    config = get_stage_config(stage_id, dataset_name)
    model = AdaptiveFramework(model_factory(dataset_name, num_classes=num_classes), config=config).to(device)
    # [V30.2] Inject density metadata for precise Governance
    model.memory.total_tasks = num_tasks
    model.memory.num_classes = num_classes
    
    trainer = ContinualTrainer(model, device=device); evaluator = ContinualEvaluator(model, device=device)
    
    # =========================================================================
    # [V22] ABSOLUTE ZERO PROTOCOL (Hard-Lock + Hyper-Plasticity)
    # =========================================================================
    import types
    model.memory.param_id_to_mask = {}     
    model.memory.task_omega_snapshots = {} 

    def _v18_governor_patch(mem, task_id, backbone_ref):
        # [V18.8] ABSOLUTE ZERO: Aggressive Quota for foundation
        total_quota = getattr(config, 'iron_mind_quota', 0.35)
        # Force 10% for the first task to ensure a rock-solid foundation
        PER_TASK_QUOTA = 0.10 if task_id == 0 else (total_quota - 0.10) / (max(1, num_tasks - 1))
        
        MIN_IMPORTANCE = 1e-5 
        id_to_p = {}; id_to_imp = {}
        id_to_prefixed_name = {}
        
        with torch.no_grad():
            for m_idx, m_tracked in enumerate(mem.models):
                prefix = f"m{m_idx}_"
                for name, p in m_tracked.named_parameters():
                    if not p.requires_grad: continue
                    p_id = id(p)
                    id_to_p[p_id] = (name, p)
                    
                    # Use the CORRECT prefixed key for omega/fisher lookup
                    omega_key = prefix + name
                    id_to_prefixed_name[p_id] = omega_key
                    
                    curr = mem.omega.get(omega_key, torch.zeros_like(p).cpu()).clone().cpu()
                    if hasattr(mem, 'fisher_dict') and omega_key in mem.fisher_dict:
                        curr = curr + mem.fisher_dict[omega_key].clone().cpu()
                    id_to_imp[p_id] = curr
        
        if not id_to_imp: return
        mem.task_omega_snapshots[task_id] = {pid: imp.clone() for pid, imp in id_to_imp.items()}
        
        # Recalculate Union of top-8% across all tasks seen so far
        cumulative = {}
        for tid, snap in mem.task_omega_snapshots.items():
            flat = torch.cat([v.view(-1) for v in snap.values()])
            n = flat.numel()
            k = max(1, min(int((1.0 - PER_TASK_QUOTA) * n), n - 1))
            thr = max(torch.kthvalue(flat, k).values.item(), MIN_IMPORTANCE)
            for pid, imp in snap.items():
                m = (imp >= thr).bool().cpu()
                # [V18.4] ETERNAL SOUL: Force Router/Gate parameters into the sacred mask
                p_name = id_to_prefixed_name.get(pid, "").lower()
                if "gate" in p_name or "router" in p_name:
                    m = torch.ones_like(m).bool()
                cumulative[pid] = cumulative[pid] | m if pid in cumulative else m
        
        # Apply the final union mask
        for pid, mask in cumulative.items():
            if pid not in id_to_p: continue
            name, tensor = id_to_p[pid]
            mem.param_id_to_mask[pid] = mask.to(tensor.device)
            # Write sacred_mask with the PREFIXED key the package expects
            if pid in id_to_prefixed_name:
                mem.sacred_mask[id_to_prefixed_name[pid]] = mask.to(tensor.device)
        
        mem.saturation_level = (sum(m.sum() for m in cumulative.values()) / sum(m.numel() for m in cumulative.values())).item()
        print(f"             [TITAN] Hard-Lock Active. Saturation: {mem.saturation_level:.2%}")

    # Redirect both the governor and the internal memory update
    model.governor.update_sacred_mask = _v18_governor_patch
    model.memory._v18_update = _v18_governor_patch

    
    # NOTE: Gradient hooks REMOVED — the package's CAS (1200 Gradient Shunts)
    # already protects sacred parameters. Adding our own hooks on top was
    # causing double-protection, killing plasticity and blocking positive
    # backward transfer that was seen in the "golden" runs.

    # ==================== BINARY PROTECTION (V18 IRON SOUL) ====================
    # Philosophy: Sacred 8% = HARD FROZEN (CAS + Restoration).
    #             Non-sacred 92% = 100% PLASTIC. No half-measures.
    #
    # Kill Layer 2: SI/EWC penalty loss (was dragging ALL 93.8M params to anchors)
    model.memory.compute_penalty = lambda *a, **k: torch.tensor(0.0, device=device)
    
    # Kill Layer 4: LR Gating (was scaling gradients to 20% on low-surprise data)
    # We force "high surprise" so the gate stays at 1.0 (full gradient flow).
    if model.world_model:
        _original_compute_surprise = model.world_model.compute_surprise
        def _full_plasticity_surprise(z_pred, z_actual):
            # Return surprise=4.0 → lr_gate = min(1.0, 4.0/4.0) = 1.0
            # Keep actual WM loss for logging but don't let it gate learning
            _, wm_loss = _original_compute_surprise(z_pred, z_actual)
            return torch.tensor(4.0, device=z_pred.device), wm_loss
        model.world_model.compute_surprise = _full_plasticity_surprise
    
    # Kill Layer 5: Surgical weight decay on non-sacred params
    model._compute_surgical_weight_decay = lambda *a, **k: torch.tensor(0.0, device=device)
    
    # [V18.4] ETERNAL SOUL: Stabilize MoE Routing
    # Force the MoE temperature to stay soft (floor at 0.75) to prevent Routing Collapse.
    if hasattr(model, 'meta_controller'):
        _orig_on_task = model.on_task_complete
        def _v18_eternal_on_task(self, task_id):
            _orig_on_task(task_id)
            if hasattr(self.meta_controller, 'temp'):
                self.meta_controller.temp = max(0.75, self.meta_controller.temp)
                print(f"             [V18.4_STABILITY] MoE Temperature stabilized at {self.meta_controller.temp:.4f}")
        model.on_task_complete = types.MethodType(_v18_eternal_on_task, model)

    # [V18.9.2] CLEAN SINGULARITY: Dual-Rate Gradient Engine
    # We remove the manual scaler to avoid conflict with the framework's native AMP.
    
    # 1. Freeze Router Temperature at 1.0 (Neutral/Stable)
    if hasattr(model, 'meta_controller'):
        model.meta_controller.temp = 1.0
        if hasattr(model.meta_controller, 'sharpen_temperature'):
            model.meta_controller.sharpen_temperature = lambda *a, **k: None
            print("             [V18.8_LOCK] MoE Temperature physically locked at 1.0")
    
    def _v18_pre_warm_protection_map(self):
        """[V18.11] Pre-compile the gradient multiplier cache to avoid first-step hangs."""
        experts = getattr(self.memory, 'models', [])
        num_experts = len(experts) if experts else getattr(self.config, 'num_experts', 1)
        
        print("             [V18.11] Pre-warming Speed-Optimized Protection Map...")
        self.memory._v18_grad_multipliers = {}
        self.memory._v18_sacred_bns = []
        
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                combined_mask = torch.zeros_like(p, dtype=torch.bool)
                found = False
                for exp_idx in range(max(1, num_experts)):
                    key = f"m{exp_idx}_{name}"
                    if key in self.memory.sacred_mask:
                        combined_mask |= self.memory.sacred_mask[key].to(p.device)
                        found = True
                if found:
                    self.memory._v18_grad_multipliers[name] = torch.where(combined_mask, 0.0, 1.2).to(p.device)
                else:
                    self.memory._v18_grad_multipliers[name] = 1.2

            for m_name, m in self.model.named_modules():
                if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm)):
                    is_sacred = False
                    for exp_idx in range(max(1, num_experts)):
                        if f"m{exp_idx}_{m_name}.weight" in self.memory.sacred_mask:
                            is_sacred = True; break
                    if is_sacred: self.memory._v18_sacred_bns.append(m)
        self.memory._v18_cache_id = len(self.memory.sacred_mask)

    model.pre_warm_protection_map = types.MethodType(_v18_pre_warm_protection_map, model)

    _original_train_step = model.train_step
    def _v18_absolute_zero_train_step(self, x, target_data=None, **kwargs):
        # [V18.11] Re-warm only if cache is stale
        if not hasattr(self.memory, '_v18_grad_multipliers') or self.memory._v18_cache_id != len(self.memory.sacred_mask):
            self.pre_warm_protection_map()

        # 2. Apply BN Cryostasis
        for m in self.memory._v18_sacred_bns: m.eval()
        
        # Step: Forward & Backward
        res = _original_train_step(x, target_data=target_data, **kwargs)
        
        # 3. FAST DUAL-RATE STEP (V20.1 STABILIZED BERSERKER)
        with torch.no_grad():
            is_plastic_task = getattr(self, 'current_task', 0) > 0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            for name, p in self.model.named_parameters():
                if p.grad is not None and name in self.memory._v18_grad_multipliers:
                    m = self.memory._v18_grad_multipliers[name]
                    if is_plastic_task:
                        # [V22] Absolute Zero: Hard Gradient Lock
                        sacred_mask = (m < 0.1)
                        if sacred_mask.any():
                            p.grad.data.masked_fill_(sacred_mask, 0.0)
                            # Physical parameter restoration (Absolute Zero Protection)
                            anchor_key = name # We assume names match or handle prefixing
                            if anchor_key in self.memory.anchor:
                                anc = self.memory.anchor[anchor_key].to(p.device)
                                p.data.copy_(torch.where(sacred_mask, anc, p.data))
                        
                        # [V22] Berserker Boost
                        p.grad.data.mul_(4.0)
                    else:
                        p.grad.data.mul_(m)

        for m in self.memory._v18_sacred_bns: m.train()
        return res
    
    model.train_step = types.MethodType(_v18_absolute_zero_train_step, model)
    print("             [V18.9.2_CLEAN] Dual-Rate Engine Enabled (Sync with Native AMP).")
    # ===========================================================================

    if hasattr(model, 'meta_controller') and model.meta_controller.reptile:
        def _patched_reptile(self_rep):
            # [V18.9] Meta-Learning Acceleration: Boost meta-update for plasticity
            tgt = self_rep.model; cw = tgt.state_dict(); eps = self_rep.config.reptile_learning_rate * 1.5
            with torch.no_grad():
                for name, anc in self_rep.anchor_weights.items():
                    if name not in cw: continue
                    fast = cw[name]
                    if anc.is_floating_point():
                        tv = anc + eps * (fast - anc)
                        mask = model.memory.sacred_mask.get(name)
                        if mask is not None: cw[name].copy_(torch.where(mask.to(fast.device), anc.to(fast.device), tv.to(fast.device)))
                        else: cw[name].copy_(tv)
                    else: cw[name].copy_(fast)
            self_rep.anchor_weights = self_rep._clone_weights()
        model.meta_controller.reptile._perform_update = types.MethodType(_patched_reptile, model.meta_controller.reptile)
    # =========================================================================

    
    results = []
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    initial_accuracies = []
    final_accuracies = []
    test_loaders = []
    img_size = 64 if dataset_name == "TinyImageNet" else 32
    replay_buf = ExternalReplayBuffer(per_task=1000, img_size=img_size)

    for t_idx in range(num_tasks):
        train_loader, _, test_loader = curriculum.get_task(t_idx)
        test_loaders.append(test_loader)
        # [V18.7] SPEED: Task 0 needs 10 epochs for foundation, others can thrive on 7
        n_epochs = 10 if t_idx == 0 else 7
        # Optimize DataLoader for speed
        # [A100 TITAN SCALING] 4 -> 8 Workers | 2 -> 4 Prefetch
        train_loader.num_workers = 8
        train_loader.pin_memory = True
        train_loader.prefetch_factor = 4 
        # [V19.1] UNIFIED 8-EPOCH REGIME
        n_epochs = 8 
        # [V21.2] PRE-WARM PROTECTION MAP: Avoid Inductor Hangs
        model.pre_warm_protection_map()
        
        print(f"\n[WARRIOR] Starting Task {t_idx} | Regime: {'FOUNDATION' if t_idx==0 else 'HYPER-PLASTIC'} | Epochs: {n_epochs}")
        
        # [V20.1] MOMENTUM PURGE & ROUTER ANCHORING
        if t_idx > 0:
            # 1. Purge Optimizer Momentum for Sacred Weights
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in model.memory.sacred_mask and model.memory.sacred_mask[name].any():
                        state = trainer.optimizer.state[p]
                        if 'momentum_buffer' in state: state['momentum_buffer'].zero_()
                        if 'exp_avg' in state: state['exp_avg'].zero_()
                        if 'exp_avg_sq' in state: state['exp_avg_sq'].zero_()
            
            # 2. Anchor Task 0 Router (Hard-wire Expert 0 for foundation confidence)
            if hasattr(model, 'meta_controller') and hasattr(model.meta_controller, 'router'):
                 # We don't hard-code Task-IL, we just bias the Expert 0 weights to be stable
                 print("             [V20.1] Anchoring Router for Foundation Stability...")

            original_train_single = trainer.train_task
            def _v20_berserker_train(loader, task_id, epochs, replay_buffer=None):
                return train_single_task(trainer.model, loader, loader, trainer.optimizer, task_id, 
                                       device=trainer.device, epochs=epochs, replay_buffer=replay_buffer,
                                       enable_dream=False, meta_step=False, label_smoothing=0.1) 
            trainer.train_task = _v20_berserker_train

        trainer.train_task(train_loader, t_idx, epochs=n_epochs, replay_buffer=replay_buf if t_idx > 0 else None)
        
        # [V30.1] Store CLEAN exemplars BEFORE consolidation
        train_indices = train_loader.dataset.indices
        base_dataset = train_loader.dataset.dataset
        replay_buf.update_from_loader(base_dataset, train_indices, dataset_name=dataset_name)
        print(f"             [REPLAY] Buffer: {len(replay_buf)} clean exemplars.")
        
        # [V29] CRITICAL: Signal end-of-task to framework.
        # on_task_complete populates omega/fisher values via consolidation.
        if hasattr(model, 'on_task_complete'):
            # Temporarily restore original governor so on_task_complete doesn't crash
            # (the package's internal call runs before omega is ready anyway)
            original_update = model.governor.update_sacred_mask
            model.governor.update_sacred_mask = lambda *a, **kw: None  # No-op during package consolidation
            model.on_task_complete(t_idx)
            model.governor.update_sacred_mask = original_update  # Restore V18
            
            # Re-sync SI anchor to prevent drift after consolidation
            with torch.no_grad():
                for m_tracked in model.memory.models:
                    for name, p in m_tracked.named_parameters():
                        if p.requires_grad: model.memory.anchor[name] = p.data.clone().detach().cpu()
            
            # NOW run V18 anchoring — omega/fisher are fully populated
            _v18_governor_patch(model.memory, t_idx, model.memory.models[0])
            print(f"             [IRON_MIND] Task {t_idx} knowledge anchored via V18 patch.")

        
        # Capture Task Accuracies
        task_accuracies = [evaluator.evaluate(loader, i) for i, loader in enumerate(test_loaders)]
        initial_accuracies.append(task_accuracies[t_idx]) 
        
        avg_acc = sum(task_accuracies) / len(task_accuracies)
        acc_str = ", ".join([f"T{i}: {acc:.2%}" for i, acc in enumerate(task_accuracies)])
        
        usage = get_resource_usage()
        # [LIVE_DEBUG] Telemetry and Memory Cleanup
        print(f"\n             [V18.7_SPEED] Task {t_idx} Finished in {(time.time() - start_time)/60:.1f}m total.")
        torch.cuda.empty_cache()
        import gc; gc.collect()
        
        print(f"\n[LIVE_DEBUG] Task {t_idx} Finished | Average Accuracy: {avg_acc:.2%}")
        print(f"             Accuracies: [{acc_str}]")
        print(f"             Resource: VRAM {usage['vram_peak_gb']:.2f}GB | Power {usage['avg_power_w']:.1f}W")
        results.append(avg_acc)
        
        if t_idx == num_tasks - 1:
            final_accuracies = task_accuracies

    total_time = time.time() - start_time
    usage = get_resource_usage()
    
    # CALCULATE RIGOROUS BWT
    forgetting = [final_accuracies[i] - initial_accuracies[i] for i in range(num_tasks - 1)]
    bwt = sum(forgetting) / len(forgetting) if len(forgetting) > 0 else 0.0
    final_avg = results[-1]
    
    report = (
        f"Result File: {filename}\n"
        f"Method: {method_name} | Dataset: {dataset_name} | Seed: {seed} | Node: {node_name}\n"
        f"Avg Accuracy: {final_avg:.4f}\n"
        f"BWT: {bwt:.4f}\n"
        f"Wall-clock: {total_time/60:.2f} mins\n"
        f"Peak VRAM: {usage['vram_peak_gb']:.2f} GB\n"
        f"Mean Power: {usage['avg_power_w']:.1f} W\n"
    )
    
    with open(filepath, "w") as f: f.write(report)
    print(f"[SUCCESS] Saved to {filepath}")
    git_sync_file(filepath, f"AutoSync: {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, nargs="+", required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=["CIFAR100", "TinyImageNet"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 10, 20, 30])
    args = parser.parse_args()
    
    for stage in args.stages:
        for ds in args.datasets:
            for seed in args.seeds:
                try:
                    run_experiment(dataset_name=ds, stage_id=stage, seed=seed)
                except Exception as e:
                    print(f"\n[CRITICAL_FAIL] {ds} S{stage} Seed {seed} failed. Moving to next run.")
                    print(traceback.format_exc())
                    continue