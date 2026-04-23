import torch
import torch.nn as nn
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig
from torchvision.models import resnet18
import time
import os

def get_memory_usage():
    # Fallback without psutil
    return 0.0 # MB

def stress_test():
    print("[START] Starting Antara V9.4 Stress Test (Infinity Horizon Check)", flush=True)
    
    # 1. Configuration: Maximize Load
    config = AdaptiveFrameworkConfig(
        model_dim=512,
        enable_consciousness=True,
        memory_type='graph',
        use_graph_memory=True,
        enable_world_model=True,
        use_moe=True,
        use_hierarchical_moe=True,
        num_domains=5, # Push MoE scaling
        experts_per_domain=4, # Total 20 experts
        input_dim=3072,
        enable_holographic_compression=True,
        feedback_buffer_size=10000, # Large buffer
        enable_health_monitor=True,
        use_ogd=True,
        ogd_max_basis_size=512, # Large basis
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    device = torch.device(config.device)
    backbone = resnet18(num_classes=100).to(device)
    
    print(f"  [INIT] Model & Framework Setup... Device: {device}", flush=True)
    agent = AdaptiveFramework(backbone, config=config)
    
    # 2. Simulated "Infinity Task" Loop
    num_tasks = 20 
    batch_size = 128 
    
    print(f"  [TEST] Running {num_tasks} Sequential Tasks | Batch Size: {batch_size}", flush=True)
    
    for t_idx in range(num_tasks):
        start_time = time.time()
        print(f"\n--- Task {t_idx+1}/{num_tasks} ---", flush=True)
        
        # Simulated high-load training
        for i in range(10): 
            x = torch.randn(batch_size, 3, 32, 32).to(device)
            y = torch.randint(0, 100, (batch_size,)).to(device)
            
            agent.train_step(x, target_data=y, task_id=t_idx)
            
            if i % 5 == 0:
                cpu_mem = get_memory_usage()
                gpu_mem = 0
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                print(f"    Step {i} | CPU: {cpu_mem:.1f}MB | GPU: {gpu_mem:.1f}MB", flush=True)

        # Force consolidation and offloading
        print("  [BRAIN] Forcing Neural Consolidation & Offloading...", flush=True)
        agent.memory.consolidate(current_step=t_idx * 10, mode='STRESS')
        
        # Verify Holographic Vault Deposit
        if hasattr(agent.memory, 'holographic_vault'):
             size = len(agent.memory.holographic_vault.vault)
             print(f"  [VAULT] Current Snapshots in Cold Storage: {size}", flush=True)

    print("\n[FINISH] Stress Test Completed Successfully!", flush=True)
    print(f"Final CPU Memory: {get_memory_usage():.1f}MB", flush=True)
    if torch.cuda.is_available():
        print(f"Final GPU Memory: {torch.cuda.memory_allocated() / (1024 * 1024):.1f}MB", flush=True)

if __name__ == "__main__":
    stress_test()
