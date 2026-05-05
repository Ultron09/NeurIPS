import os
import sys
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

# [V25.4] RESTORED STATIC MOE DISPATCHER
if hasattr(moe_mod, 'SparseMoE'):
    print("             [V25.4] Deploying Fixed Static MoE Dispatcher...")
    def _unified_sparse_fwd(self, x, task_id=None, consciousness_state=None, *args, **kwargs):
        # 1. Static Routing Logic (Fused)
        logits = self.gate(x, consciousness_state=consciousness_state)
        if isinstance(logits, tuple): logits = logits[0]
        
        # [V25.4] CRITICAL: Hard argmax for static kernels
        with torch.no_grad():
            indices = torch.argmax(logits, dim=-1)
            
        # 2. Static Dispatch (No None checks, No mask.any breaks)
        if not hasattr(self, '_v24_out_dim'):
            with torch.no_grad():
                test_out = self.experts[0](x[:1], task_id=task_id)
                if isinstance(test_out, tuple): test_out = test_out[0]
                self._v24_out_dim = test_out.shape[1]
        
        outputs = None
        for i in range(self.num_experts):
            mask = (indices == i)
            if not mask.any(): continue
            
            expert_out = self.experts[i](x[mask], task_id=task_id)
            if isinstance(expert_out, tuple): expert_out = expert_out[0]
            
            if outputs is None:
                outputs = torch.zeros(x.size(0), self._v24_out_dim, device=x.device, dtype=expert_out.dtype)
            
            outputs[mask] = expert_out
            
        return outputs, indices

    moe_mod.SparseMoE.forward = _unified_sparse_fwd
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.suppress_errors = True

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
    def __init__(self, per_task=1000, img_size=32):
        self.per_task = per_task
        self.img_size = img_size
        self.x_data = []; self.y_data = []
        self._aug = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ])
        self._cache_x = None; self._cache_y = None

    def update_from_loader(self, dataset, indices, dataset_name="CIFAR100"):
        import copy
        from torch.utils.data import DataLoader, Subset
        clean_dataset = copy.copy(dataset)
        if hasattr(clean_dataset, 'transform'):
            if dataset_name == "CIFAR100": norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            else: norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            clean_dataset.transform = transforms.Compose([transforms.ToTensor(), norm])
        clean_loader = DataLoader(Subset(clean_dataset, indices), batch_size=128, shuffle=False)
        all_x, all_y = [], []
        for x, y in clean_loader:
            all_x.append(x.cpu()); all_y.append(y.cpu())
        if not all_x: return
        all_x = torch.cat(all_x); all_y = torch.cat(all_y)
        sel_idx = torch.randperm(len(all_x))[:self.per_task]
        self.x_data.append(all_x[sel_idx]); self.y_data.append(all_y[sel_idx])
        self._cache_x = None; self._cache_y = None

    def __len__(self):
        return sum(x.size(0) for x in self.x_data)

    def sample(self, batch_size):
        if self._cache_x is None:
            self._cache_x = torch.cat(self.x_data); self._cache_y = torch.cat(self.y_data)
        indices = torch.randperm(len(self._cache_x))[:batch_size]
        return self._aug(self._cache_x[indices]), self._cache_y[indices]

class ContinualTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model; self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=getattr(model.config, 'learning_rate', 5e-4))
    def train_task(self, loader, t_idx, epochs=10, replay_buffer=None):
        if hasattr(self, 'optimizer'):
            del self.optimizer
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=getattr(self.model.config, 'learning_rate', 5e-4))
        res = train_single_task(self.model, loader, loader, self.optimizer, t_idx, device=self.device, epochs=epochs, replay_buffer=replay_buffer)
        print(f"             [DEBUG] ContinualTrainer.train_task returning for Task {t_idx}.")
        return res

class ContinualEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model; self.device = device
    def evaluate(self, loader, t_idx):
        self.model.eval()
        correct = 0; total = 0
        with torch.inference_mode():
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
    if stage_id == 7: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=1.0, use_ogd=True, novelty_z_threshold=1.2, adaptation_threshold=0.05, enable_consciousness=True, use_reptile=True, iron_mind_quota=0.35, use_learned_optimizer=False)
    return AdaptiveFrameworkConfig(**base_params)

def run_experiment(dataset_name="CIFAR100", stage_id=7, seed=42):
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
    # [V25.13] INSTANCE LEVEL KILL: Ensure even bound methods are nullified
    AdaptiveFramework._rebuild_restoration_cache = lambda self: None
    model = AdaptiveFramework(model_factory(dataset_name, num_classes=num_classes), config=config).to(device)
    model.on_task_complete = lambda task_id: None 
    print(f"             [DEBUG] Model device: {next(model.parameters()).device}")
    
    print("             [DEBUG] Initializing Trainer...")
    trainer = ContinualTrainer(model, device=device); evaluator = ContinualEvaluator(model, device=device)
    print("             [DEBUG] Trainer Ready.")
    
    # [V22] PATCHES
    model.memory.param_id_to_mask = {}; model.memory.task_omega_snapshots = {} 

    def _v18_governor_patch(mem, task_id, backbone_ref):
        total_quota = getattr(config, 'iron_mind_quota', 0.35)
        PER_TASK_QUOTA = 0.10 if task_id == 0 else (total_quota - 0.10) / (max(1, num_tasks - 1))
        id_to_p = {}; id_to_imp = {}; id_to_prefixed_name = {}
        with torch.no_grad():
            for m_idx, m_tracked in enumerate(mem.models):
                prefix = f"m{m_idx}_"
                for name, p in m_tracked.named_parameters():
                    if not p.requires_grad: continue
                    omega_key = prefix + name
                    id_to_p[id(p)] = (name, p); id_to_prefixed_name[id(p)] = omega_key
                    curr = mem.omega.get(omega_key, torch.zeros_like(p).cpu()).clone().cpu()
                    if hasattr(mem, 'fisher_dict') and omega_key in mem.fisher_dict: curr += mem.fisher_dict[omega_key].clone().cpu()
                    id_to_imp[id(p)] = curr
        if not id_to_imp: return
        mem.task_omega_snapshots[task_id] = {pid: imp.clone() for pid, imp in id_to_imp.items()}
        cumulative = {}
        device = next(backbone_ref.parameters()).device
        for snap in mem.task_omega_snapshots.values():
            thr = 1e-4 # [V25.9] Fixed Absolute Threshold for Stability
            for pid, imp in snap.items():
                m = (imp.to(device) >= thr).bool().cpu()
                if "gate" in id_to_prefixed_name.get(pid, "").lower(): m = torch.ones_like(m).bool()
                cumulative[pid] = cumulative[pid] | m if pid in cumulative else m
        for pid, mask in cumulative.items():
            if pid not in id_to_p: continue
            mem.param_id_to_mask[pid] = mask.to(id_to_p[pid][1].device)
            mem.sacred_mask[id_to_prefixed_name[pid]] = mask.to(id_to_p[pid][1].device)
        print(f"             [TITAN] Hard-Lock Active. Saturation: {sum(m.sum() for m in cumulative.values())/sum(m.numel() for m in cumulative.values()):.2%}")

    model.governor.update_sacred_mask = _v18_governor_patch
    model.memory._v18_update = _v18_governor_patch
    model.memory.compute_penalty = lambda *a, **k: torch.tensor(0.0, device=device)
    
    @torch._dynamo.disable
    def _v18_pre_warm_protection_map(self):
        experts = getattr(self.memory, 'models', [])
        num_experts = len(experts) if experts else 1
        self.memory._v24_flat_grad_logic = []
        self.memory._v18_sacred_bns = []
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                mask = torch.zeros_like(p, dtype=torch.bool); found = False
                for exp_idx in range(num_experts):
                    key = f"m{exp_idx}_{name}"
                    if key in self.memory.sacred_mask: mask |= self.memory.sacred_mask[key].to(p.device); found = True
                mult = torch.where(mask, 0.0, 1.2).to(p.device) if found else torch.full_like(p, 1.2)
                self.memory._v24_flat_grad_logic.append((p, mult, self.memory.anchor.get(name, p.data.clone()).to(p.device), mask))
            for n, m in self.model.named_modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    if any(f"m{i}_{n}.weight" in self.memory.sacred_mask for i in range(num_experts)): self.memory._v18_sacred_bns.append(m)
        self.memory._v18_cache_id = len(self.memory.sacred_mask)

    model.pre_warm_protection_map = types.MethodType(_v18_pre_warm_protection_map, model)
    _orig_train_step = model.train_step
    def _v18_absolute_zero_train_step(self, x, target_data=None, **kwargs):
        if not hasattr(self.memory, '_v24_flat_grad_logic') or self.memory._v18_cache_id != len(self.memory.sacred_mask): self.pre_warm_protection_map()
        for m in self.memory._v18_sacred_bns: m.eval()
        res = _orig_train_step(x, target_data=target_data, **kwargs)
        with torch.no_grad():
            is_plastic = getattr(self, 'current_task', 0) > 0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            for p, mult, anc, mask in self.memory._v24_flat_grad_logic:
                if p.grad is not None:
                    if is_plastic: p.grad.data.masked_fill_(mask, 0.0); p.data.copy_(torch.where(mask, anc, p.data)); p.grad.data.mul_(4.0)
                    else: p.grad.data.mul_(mult)
        for m in self.memory._v18_sacred_bns: m.train()
        import sys
        sys.stderr.write(".")
        sys.stderr.flush()
        return res
    model.train_step = types.MethodType(_v18_absolute_zero_train_step, model)

    results = []; start_time = time.time(); test_loaders = []
    replay_buf = ExternalReplayBuffer(per_task=1000, img_size=64 if dataset_name == "TinyImageNet" else 32)

    for t_idx in range(num_tasks):
        train_loader, _, test_loader = curriculum.get_task(t_idx)
        test_loaders.append(test_loader)
        train_loader.num_workers = 0; train_loader.pin_memory = True; train_loader.prefetch_factor = None
        
        print("             [DEBUG] Pre-warming Protection Map...")
        model.pre_warm_protection_map()
        print(f"\n[WARRIOR] Starting Task {t_idx} | Epochs: 8")
        
        if t_idx > 0:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in model.memory.sacred_mask and model.memory.sacred_mask[name].any():
                        for k in ['momentum_buffer', 'exp_avg', 'exp_avg_sq']:
                            if k in trainer.optimizer.state[p]: trainer.optimizer.state[p][k].zero_()
            trainer.train_task = lambda loader, task_id, epochs, replay_buffer=None: train_single_task(trainer.model, loader, loader, trainer.optimizer, task_id, device=trainer.device, epochs=epochs, replay_buffer=replay_buffer, label_smoothing=0.1)

        trainer.train_task(train_loader, t_idx, epochs=8, replay_buffer=replay_buf if t_idx > 0 else None)
        print("             [DEBUG] Training Task Complete. Updating Replay Buffer...", flush=True)
        replay_buf.update_from_loader(train_loader.dataset.dataset, train_loader.dataset.indices, dataset_name=dataset_name)
        
        print("             [DEBUG] Replay Buffer Updated. Anchoring Parameters...", flush=True)
        # [V25.12] Manual Anchoring only — bypasses library's unstable on_task_complete
        with torch.no_grad():
            for m_tr in model.memory.models:
                for n, p in m_tr.named_parameters():
                    if p.requires_grad: model.memory.anchor[n] = p.data.clone().detach().cpu()
        
        print("             [DEBUG] Parameters Anchored. Patching Governor...", flush=True)
        _v18_governor_patch(model.memory, t_idx, model.memory.models[0])

        print("             [DEBUG] Governor Patched. Evaluating Performance...", flush=True)
        task_accs = [evaluator.evaluate(l, i) for i, l in enumerate(test_loaders)]
        print(f"             [LIVE] Avg Accuracy: {sum(task_accs)/len(task_accs):.2%}", flush=True)
        torch.cuda.empty_cache(); gc.collect()
        print("             [DEBUG] Task Transition Complete.", flush=True)

    report = f"Method: ANTARA_S{stage_id} | Seed: {seed}\nAvg Accuracy: {sum(task_accs)/len(task_accs):.4f}\n"
    with open(filepath, "w") as f: f.write(report)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 10, 20, 30])
    args = parser.parse_args()
    for stage in args.stages:
        for seed in args.seeds:
            run_experiment(stage_id=stage, seed=seed)