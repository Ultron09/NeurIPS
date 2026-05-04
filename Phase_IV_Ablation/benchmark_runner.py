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
from torchvision.models.resnet import ResNet, BasicBlock
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

    def __len__(self):
        return sum(x.size(0) for x in self.x_data)

    def sample(self, batch_size):
        """Random sample across all stored tasks with on-the-fly augmentation."""
        all_x = torch.cat(self.x_data)
        all_y = torch.cat(self.y_data)
        indices = torch.randperm(len(all_x))[:batch_size]
        batch_x = all_x[indices]
        # Apply augmentation on-the-fly for diversity
        batch_x = self._aug(batch_x)
        return batch_x, all_y[indices]


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
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
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
        "learning_rate": 5e-4, "use_gradient_centralization": True, "use_lookahead": True,
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
    if stage_id == 7: return AdaptiveFrameworkConfig(**base_params, use_moe=True, use_hierarchical_moe=True, si_lambda=2.5, enable_consciousness=True, use_reptile=True, enable_world_model=True, iron_mind_quota=0.35, enable_dreaming=True, dream_interval=5)
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = os.path.join(root_path, "data")
    if dataset_name == "CIFAR100": curriculum = SplitCIFAR100(root=data_path); num_classes = 100; num_tasks = 10
    else: curriculum = SplitTinyImageNet(root=data_path); num_classes = 200; num_tasks = 10
    
    config = get_stage_config(stage_id, dataset_name)
    model = AdaptiveFramework(model_factory(dataset_name, num_classes=num_classes), config=config).to(device)
    # [V30.2] Inject density metadata for precise Governance
    model.memory.total_tasks = num_tasks
    model.memory.num_classes = num_classes
    
    trainer = ContinualTrainer(model, device=device); evaluator = ContinualEvaluator(model, device=device)
    
    # =========================================================================
    # [V18] IRON SOUL MONKEY-PATCH (Knowledge Anchoring)
    # =========================================================================
    print("             [SYSTEM] Injecting Neuro-Stability V18 IRON SOUL...")
    import types
    model.memory.param_id_to_mask = {}     
    model.memory.task_omega_snapshots = {} 

    def _v18_update(mem, task_id, backbone_ref):
        PER_TASK_QUOTA = 0.08 
        MIN_IMPORTANCE = 1e-5 
        id_to_p = {}; id_to_imp = {}
        with torch.no_grad():
            for m_tracked in mem.models:
                for name, p in m_tracked.named_parameters():
                    if not p.requires_grad: continue
                    p_id = id(p); id_to_p[p_id] = (name, p)
                    curr = mem.omega.get(name, torch.zeros_like(p).cpu()).clone()
                    if name in mem.fisher_dict: curr = curr + mem.fisher_dict[name].cpu()
                    id_to_imp[p_id] = curr
        if not id_to_imp: return
        mem.task_omega_snapshots[task_id] = {pid: imp.clone() for pid, imp in id_to_imp.items()}
        cumulative = {}
        for tid, snap in mem.task_omega_snapshots.items():
            flat = torch.cat([v.view(-1) for v in snap.values()])
            n = flat.numel()
            k = max(1, min(int((1.0 - PER_TASK_QUOTA) * n), n - 1))
            thr = max(torch.kthvalue(flat, k).values.item(), MIN_IMPORTANCE)
            for pid, imp in snap.items():
                m = (imp >= thr).bool()
                cumulative[pid] = cumulative[pid] | m if pid in cumulative else m
        
        # Hard-lock FC head rows
        fc = getattr(backbone_ref, 'fc', None)
        if fc is not None:
            fc_w_id = id(fc.weight)
            if fc_w_id not in cumulative: cumulative[fc_w_id] = torch.zeros(fc.weight.shape, dtype=torch.bool)
            for tid in mem.task_omega_snapshots:
                s, e = tid * 10, min((tid + 1) * 10, fc.weight.shape[0])
                cumulative[fc_w_id][s:e, :] = True
                if fc.bias is not None:
                    fc_b_id = id(fc.bias)
                    if fc_b_id not in cumulative: cumulative[fc_b_id] = torch.zeros(fc.bias.shape, dtype=torch.bool)
                    cumulative[fc_b_id][s:e] = True
        
        mem.param_id_to_mask = {}
        all_names = {}
        for m_tracked in mem.models:
            for name, p in m_tracked.named_parameters(): all_names[id(p)] = name
        
        prot = 0; tot = 0
        for pid, mask in cumulative.items():
            if pid not in id_to_p: continue
            tensor = id_to_p[pid][1]
            mem.param_id_to_mask[pid] = mask.to(tensor.device)
            prot += mask.sum().item(); tot += mask.numel()
            if pid in all_names: mem.sacred_mask[all_names[pid]] = mask
        mem.saturation_level = prot / tot if tot > 0 else 0.0
        print(f"             [SENTIENT] Sacred Mask Updated. Global Saturation: {mem.saturation_level:.2%}")

    model.memory._v18_update = _v18_update
    
    def _make_hook(p_id, mem):
        def hook(grad):
            m = mem.param_id_to_mask.get(p_id)
            if m is not None: return grad * (~m.to(grad.device))
            return grad
        return hook

    for tracked_model in model.memory.models:
        for p in tracked_model.parameters():
            if p.requires_grad: p.register_hook(_make_hook(id(p), model.memory))

    if hasattr(model, 'meta_controller') and model.meta_controller.reptile:
        def _patched_reptile(self_rep):
            tgt = self_rep.model; cw = tgt.state_dict(); eps = self_rep.config.reptile_learning_rate
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
    img_size = 64 if dataset_name == "TinyImageNet" else 32
    replay_buf = ExternalReplayBuffer(per_task=1000, img_size=img_size)

    for t_idx in range(num_tasks):
        train_loader, _, _ = curriculum.get_task(t_idx)
        test_loaders = [curriculum.get_task(i)[2] for i in range(t_idx + 1)]
        trainer.train_task(train_loader, t_idx, epochs=10, replay_buffer=replay_buf if t_idx > 0 else None)
        
        # [V30.1] Store CLEAN exemplars BEFORE consolidation
        # We pass the underlying dataset and indices to avoid augmented samples
        train_indices = train_loader.dataset.indices
        base_dataset = train_loader.dataset.dataset
        replay_buf.update_from_loader(base_dataset, train_indices, dataset_name=dataset_name)
        print(f"             [REPLAY] Buffer: {len(replay_buf)} clean exemplars.")
        
        # [V29] CRITICAL: Signal end-of-task to framework.
        # This triggers KnowledgeGovernor (Iron Mind) to lock Task N knowledge.
        if hasattr(model, 'on_task_complete'):
            model.on_task_complete(t_idx)
            # Re-sync SI anchor to prevent drift after consolidation
            with torch.no_grad():
                for m_tracked in model.memory.models:
                    for name, p in m_tracked.named_parameters():
                        if p.requires_grad: model.memory.anchor[name] = p.data.clone().detach().cpu()
            
            # [V18] Run absolute knowledge anchoring
            model.memory._v18_update(model.memory, t_idx, model.memory.models[0])
            print(f"             [IRON_MIND] Task {t_idx} knowledge anchored.")
        
        # Capture Task Accuracies
        task_accuracies = [evaluator.evaluate(loader, i) for i, loader in enumerate(test_loaders)]
        initial_accuracies.append(task_accuracies[t_idx]) 
        
        avg_acc = sum(task_accuracies) / len(task_accuracies)
        acc_str = ", ".join([f"T{i}: {acc:.2%}" for i, acc in enumerate(task_accuracies)])
        
        usage = get_resource_usage()
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
    args = parser.parse_args()
    
    SEEDS = [42, 10, 20, 30]
    DATASETS = ["CIFAR100", "TinyImageNet"]
    
    for stage in args.stages:
        for seed in SEEDS:
            for ds in DATASETS:
                try:
                    run_experiment(dataset_name=ds, stage_id=stage, seed=seed)
                except Exception as e:
                    print(f"\n[CRITICAL_FAIL] {ds} S{stage} Seed {seed} failed. Moving to next run.")
                    print(traceback.format_exc())
                    continue