import torch
import torch.nn as nn
from ablation_runner import AblationOrchestrator
from airborne_antara import AdaptiveFrameworkConfig
from torchvision.models import resnet18

def model_factory():
    return resnet18(num_classes=100)

if __name__ == "__main__":
    config = AdaptiveFrameworkConfig(
        enable_consciousness=True,
        memory_type='graph',
        use_graph_memory=True, # Explicitly enable graph memory
        enable_holographic_compression=True,
        enable_world_model=True, # [V9.0] Synthetic Intuition
        use_moe=True, # [V7.1] Required gate for MoE
        use_hierarchical_moe=True,
        input_dim=3072, # CIFAR-100 flattened (3*32*32) for MoE gating
        enable_health_monitor=True,
        health_check_interval=100,
        use_ogd=True,
        ogd_max_basis_size=256,
        ewc_lambda=1000,
        use_gradient_centralization=True,
        use_lookahead=True
    )
    
    orchestrator = AblationOrchestrator(
        model_factory=model_factory, 
        config_name="Vanguard_Full",
        epochs=50,
        patience=5
    )
    
    orchestrator.run_experiment(config)
