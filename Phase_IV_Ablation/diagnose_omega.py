"""
DIAGNOSTIC: Why is mem.omega empty?
Run this ONCE to understand what the memory system actually stores.
Expected runtime: ~12 minutes (Task 0 only).
"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Phase_IV_Ablation.benchmark_runner import (
    get_stage_config, model_factory, SplitCIFAR100,
    ContinualTrainer, ExternalReplayBuffer
)
from airborne_antara import AdaptiveFramework

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = os.path.join(os.path.dirname(__file__), "..", "data")

config = get_stage_config(7, "CIFAR100")
model = AdaptiveFramework(model_factory("CIFAR100", num_classes=100), config=config).to(device)
model.memory.total_tasks = 10
model.memory.num_classes = 100

curriculum = SplitCIFAR100(root=data_path)
trainer = ContinualTrainer(model, device=device)

# Train Task 0
train_loader, _, _ = curriculum.get_task(0)
trainer.train_task(train_loader, 0, epochs=2)  # Only 2 epochs for speed

print("\n" + "="*70)
print("DIAGNOSTIC: Memory State BEFORE on_task_complete")
print("="*70)

mem = model.memory

# Check omega
print(f"\n[1] type(mem.omega) = {type(mem.omega)}")
print(f"    len(mem.omega)  = {len(mem.omega)}")
if len(mem.omega) > 0:
    keys = list(mem.omega.keys())[:5]
    print(f"    First 5 keys: {keys}")
    for k in keys[:3]:
        v = mem.omega[k]
        print(f"      '{k}': shape={v.shape}, min={v.min():.6e}, max={v.max():.6e}, mean={v.mean():.6e}")
else:
    print("    >>> OMEGA IS EMPTY <<<")

# Check fisher_dict
print(f"\n[2] hasattr(mem, 'fisher_dict') = {hasattr(mem, 'fisher_dict')}")
if hasattr(mem, 'fisher_dict'):
    print(f"    type(mem.fisher_dict) = {type(mem.fisher_dict)}")
    print(f"    len(mem.fisher_dict)  = {len(mem.fisher_dict)}")
    if len(mem.fisher_dict) > 0:
        keys = list(mem.fisher_dict.keys())[:5]
        print(f"    First 5 keys: {keys}")
        for k in keys[:3]:
            v = mem.fisher_dict[k]
            print(f"      '{k}': shape={v.shape}, min={v.min():.6e}, max={v.max():.6e}, mean={v.mean():.6e}")

# Check sacred_mask
print(f"\n[3] hasattr(mem, 'sacred_mask') = {hasattr(mem, 'sacred_mask')}")
if hasattr(mem, 'sacred_mask'):
    print(f"    len(mem.sacred_mask) = {len(mem.sacred_mask)}")

# Check what attributes mem has that look relevant
print(f"\n[4] All mem attributes containing 'omega', 'fisher', 'importance', 'mask', 'si':")
for attr in sorted(dir(mem)):
    if any(kw in attr.lower() for kw in ['omega', 'fisher', 'importance', 'mask', 'si', 'sacred', 'anchor']):
        obj = getattr(mem, attr)
        if isinstance(obj, dict):
            print(f"    mem.{attr} -> dict, len={len(obj)}")
        elif isinstance(obj, torch.Tensor):
            print(f"    mem.{attr} -> Tensor, shape={obj.shape}")
        elif callable(obj):
            print(f"    mem.{attr} -> method")
        else:
            print(f"    mem.{attr} -> {type(obj).__name__}: {obj}")

# Now run on_task_complete
print("\n" + "="*70)
print("Running model.on_task_complete(0)...")
print("="*70)
model.on_task_complete(0)

print("\n" + "="*70)
print("DIAGNOSTIC: Memory State AFTER on_task_complete")
print("="*70)

print(f"\n[5] len(mem.omega) = {len(mem.omega)}")
if len(mem.omega) > 0:
    keys = list(mem.omega.keys())[:5]
    print(f"    First 5 keys: {keys}")
    for k in keys[:3]:
        v = mem.omega[k]
        print(f"      '{k}': shape={v.shape}, min={v.min():.6e}, max={v.max():.6e}, mean={v.mean():.6e}")
    
    # Check how many are above threshold
    all_vals = torch.cat([v.view(-1) for v in mem.omega.values()])
    print(f"\n    Total omega values: {all_vals.numel()}")
    print(f"    Values > 1e-5: {(all_vals > 1e-5).sum().item()} ({(all_vals > 1e-5).float().mean():.2%})")
    print(f"    Values > 1e-3: {(all_vals > 1e-3).sum().item()} ({(all_vals > 1e-3).float().mean():.2%})")
    print(f"    Values > 0.01: {(all_vals > 0.01).sum().item()} ({(all_vals > 0.01).float().mean():.2%})")
    print(f"    Max value: {all_vals.max():.6e}")
else:
    print("    >>> OMEGA IS STILL EMPTY AFTER CONSOLIDATION <<<")

print(f"\n[6] Fisher after consolidation:")
if hasattr(mem, 'fisher_dict') and len(mem.fisher_dict) > 0:
    all_fisher = torch.cat([v.view(-1).cpu() for v in mem.fisher_dict.values()])
    print(f"    Total fisher values: {all_fisher.numel()}")
    print(f"    Values > 1e-5: {(all_fisher > 1e-5).sum().item()}")
    print(f"    Max value: {all_fisher.max():.6e}")
else:
    print("    >>> FISHER IS EMPTY <<<")

# Check models tracked
print(f"\n[7] mem.models: {len(mem.models)} models tracked")
for i, m in enumerate(mem.models):
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"    Model {i}: {type(m).__name__}, {n_params:,} trainable params")
    first_names = [n for n, _ in list(m.named_parameters())[:3]]
    print(f"      First 3 param names: {first_names}")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
