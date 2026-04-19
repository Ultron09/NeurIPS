import torch
import torch.nn as nn
from ablation_runner import AblationOrchestrator
from airborne_antara import AdaptiveFrameworkConfig
from torchvision.models import resnet18

def model_factory():
    return resnet18(num_classes=100)

if __name__ == "__main__":
    config = AdaptiveFrameworkConfig(
        enable_consciousness=False,
        memory_type='graph',
        use_hierarchical_moe=True,
        ewc_lambda=5000
    )
    
    orchestrator = AblationOrchestrator(
        model_factory=model_factory, 
        config_name="Ablated_Consciousness",
        epochs=50,
        patience=5
    )
    
    orchestrator.run_experiment(config)
