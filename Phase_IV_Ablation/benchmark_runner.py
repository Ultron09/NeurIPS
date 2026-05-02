import argparse
import time
import sys
import os
import torch
import copy
from torchvision.models import resnet18

# Path setup to import from other phases and local framework
for d in ['Phase_I_Curriculum', 'Phase_II_Baselines', 'Phase_III_Metrics']:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), d))

from dataset import SplitCIFAR100, SplitMNIST, set_seed
from baselines import EWC, ExperienceReplay, AGEM, DERPlus, HAT, iCaRL
from metrics import MetricsEngine
from trainer import train_single_task
from evaluation import evaluate_suite
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig

def setup_compute(device_str):
    if device_str == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")

def run_experiment(method_name, device_str, wandb_sync=False, project="NeurIPS", entity=None, suffix="", seed=42, dataset_name="CIFAR100"):
    set_seed(seed)
    device = setup_compute(device_str)
    full_method_name = f"{method_name}{suffix}_seed{seed}"

    print(f"\n[NEURIPS GAUNTLET] Executing Branch: {full_method_name}")
    print(f"  [SYSTEM] Active Compute Device: {device}")

    if wandb_sync:
        import wandb
        wandb.init(project=project, entity=entity, name=full_method_name, config={
            "method": method_name,
            "seed": seed,
            "device": str(device)
        })

    def model_factory():
        return resnet18(num_classes=100)

    if dataset_name == "CIFAR100":
        curriculum = SplitCIFAR100(pin_memory=(device.type == "cuda"))
        num_tasks = 10
    else:
        curriculum = SplitMNIST(pin_memory=(device.type == "cuda"))
        num_tasks = 5
    model = model_factory().to(device)

    ewc_module = None
    replay_buffer = None
    agem_module = None
    der_module = None
    hat_module = None
    icarl_module = None
    config = None

    if method_name == "ANTARA_FULL":
        config = AdaptiveFrameworkConfig(
            model_dim=256,
            num_experts=16,            # [KILLSHOT] Increased expert density for O(1) scaling
            top_k_experts=2,
            use_moe=True,
            use_hierarchical_moe=True,
            enable_consciousness=False, # [KILLSHOT] Enable System 2 Introspection
            enable_world_model=False,   # [KILLSHOT] Predictive latent foresight
            use_ogd=True,
            ogd_max_basis_size=1024,
            iron_mind_quota=0.25,      # [KILLSHOT] Optimized for maximum plasticity
            moe_temperature=1.0,
            moe_temp_decay=0.90,       # [KILLSHOT] Slower sharpening for better generalization
            input_dim=3072,
            learning_rate=2e-3,
            ewc_lambda=0.0,
            si_lambda=1.5,             # [KILLSHOT] Stronger synaptic stability
            use_reptile=True,
            reptile_learning_rate=0.1,
            use_learned_optimizer=False, # [KILLSHOT] Dynamic meta-optimization
            novelty_z_threshold=1.1,    # [KILLSHOT] More sensitive novelty detection
            adaptation_threshold=0.04,
            use_gradient_centralization=True,
            use_lookahead=True
        )
        model = AdaptiveFramework(model, config=config, device=device)
    elif method_name == "ANTARA_RGW_ONLY":
        config = AdaptiveFrameworkConfig(
            model_dim=256,
            num_experts=10,
            top_k_experts=2,
            use_moe=True,
            use_hierarchical_moe=True,
            use_ogd=False,
            input_dim=3072,
            learning_rate=2e-3,
            ewc_lambda=0.0,
            si_lambda=1.0,
            use_reptile=True,
            reptile_learning_rate=0.1,
            use_learned_optimizer=False,
            novelty_z_threshold=1.2,
            adaptation_threshold=0.05,
            use_gradient_centralization=True,
            use_lookahead=True
        )
        model = AdaptiveFramework(model, config=config, device=device)
    elif method_name == "ANTARA_OGD_ONLY":
        config = AdaptiveFrameworkConfig(
            model_dim=256,
            num_experts=10,
            top_k_experts=2,
            use_moe=True,
            use_hierarchical_moe=True,
            use_ogd=True,
            ogd_max_basis_size=1024,
            iron_mind_quota=0.15,
            input_dim=3072,
            learning_rate=2e-3,
            ewc_lambda=0.0,
            si_lambda=1.0,
            use_reptile=False,
            use_learned_optimizer=False,
            novelty_z_threshold=1.2,
            adaptation_threshold=0.05,
            use_gradient_centralization=True,
            use_lookahead=True
        )
        model = AdaptiveFramework(model, config=config, device=device)
    elif method_name == "EWC":
        ewc_module = EWC(model, lambda_factor=5000)
    elif method_name == "REPLAY":
        replay_buffer = ExperienceReplay(buffer_size=2000)
    elif method_name == "A-GEM":
        agem_module = AGEM(model, buffer_size=2000)
    elif method_name == "DER++":
        der_module = DERPlus(model, buffer_size=2000, alpha=0.1)
    elif method_name == "HAT":
        hat_module = HAT(model, num_tasks=num_tasks).to(device)
    elif method_name == "iCaRL":
        icarl_module = iCaRL(model, buffer_size=2000, num_classes=100).to(device)
    elif method_name == "NAIVE":
        pass
    else:
        raise ValueError(f"Invalid experiment method: {method_name}")

    is_antara = method_name.startswith("ANTARA")

    if is_antara:
        print("  [SYSTEM] Initializing Neuro-Stability V15 IRON MIND (Native Package Edition)...")

    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================
    optimizer = None if is_antara else torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    metrics   = MetricsEngine(num_tasks=num_tasks, config_name=full_method_name)
    total_start_time  = time.time()
    task_step_times   = []

    for t_idx in range(num_tasks):
        train_loader, val_loader, _ = curriculum.get_task(t_idx)

        avg_step_time = train_single_task(
            model, train_loader, val_loader, optimizer, t_idx,
            device=device, ewc_module=ewc_module,
            agem_module=agem_module, replay_buffer=replay_buffer,
            der_module=der_module, hat_module=hat_module,
            epochs=10
        )
        task_step_times.append(avg_step_time)

        if method_name == "EWC":
            ewc_module.save_task_weights(train_loader, device=device)
        elif method_name == "HAT":
            hat_module.update_cumulative_mask(t_idx)
        elif method_name == "iCaRL":
            for c in curriculum.task_classes[t_idx]:
                icarl_module.update_exemplars(train_loader.dataset, c, device=device)
            icarl_module.compute_means(device=device)

        if is_antara:
            model.on_task_complete(t_idx)

        evaluate_suite(model, curriculum, t_idx, metrics, device=device, hat_module=hat_module)

    # =========================================================================
    # FINALIZE
    # =========================================================================
    total_duration = time.time() - total_start_time
    metrics.avg_step_time_ms        = (sum(task_step_times) / len(task_step_times))
    metrics.total_compute_time_sec  = total_duration
    if device.type == "cuda":
        metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    results_path = f"results/{full_method_name}_metrics.json"
    metrics.save_results(results_path)
    metrics.plot_heatmap(f"results/{full_method_name}_heatmap.png")
    metrics.generate_summary_report()

    print(f"\n[GAUNTLET COMPLETE] Method: {full_method_name}")
    print(f"  Total Duration: {total_duration/60:.2f} minutes")
    print(f"  Avg Step Time: {sum(task_step_times)/len(task_step_times):.4f}s")

    if wandb_sync:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["ANTARA_FULL", "ANTARA_RGW_ONLY", "ANTARA_OGD_ONLY", "EWC", "REPLAY", "A-GEM", "DER++", "HAT", "NAIVE", "iCaRL"])
    parser.add_argument("--dataset", type=str, default="CIFAR100", choices=["CIFAR100", "MNIST"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="NeurIPS")
    parser.add_argument("--entity", type=str, default="ultron09-airbornehrs")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    run_experiment(args.method, args.device, args.wandb, args.project, args.entity, args.suffix, args.seed, args.dataset)