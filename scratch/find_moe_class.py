import airborne_antara.moe as m
import inspect
import torch

classes = ['AdaptiveExpertBlock', 'ExpertBlock', 'SparseMoE', 'HierarchicalMoE']
for n in classes:
    try:
        cls = getattr(m, n)
        src = inspect.getsource(cls)
        if '.item()' in src:
            print(f"FOUND: {n}")
    except:
        pass
