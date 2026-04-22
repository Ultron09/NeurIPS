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
        use_hierarchical_moe=True,
        enable_health_monitor=True,
        use_ogd=True,
        ogd_max_basis_size=256,
        ewc_lambda=1000
    )
    
    orchestrator = AblationOrchestrator(
        model_factory=model_factory, 
        config_name="Vanguard_Full",
        epochs=50,
        patience=5
    )
    
    orchestrator.run_experiment(config)
