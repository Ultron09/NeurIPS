import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import argparse
import copy
import json
from typing import Optional

# Path Management
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub_dir in ['Phase_I_Curriculum', 'Phase_II_Baselines', 'Phase_III_Metrics']:
    path = os.path.join(ROOT_DIR, sub_dir)
    if path not in sys.path:
        sys.path.append(path)

from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig
from curriculum import SplitCIFAR100, set_seed
from metrics_engine import MetricsEngine
from baselines import EWC

class AblationOrchestrator:
    """
    Standardizes the Architectural Autopsy for NeurIPS.
    Features Self-Healing Checkpointing for Resume-on-Failure.
    """
    def __init__(self, model_factory, epochs=50, patience=5, seed=42, device='cuda', config_name="Default"):
        self.model_factory = model_factory
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = device
        self.config_name = config_name
        
        set_seed(self.seed)
        self.curriculum = SplitCIFAR100(seed=self.seed)
        
        self.checkpoint_dir = os.path.join(ROOT_DIR, 'Phase_IV_Ablation', 'results', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def get_checkpoint_paths(self):
        base = os.path.join(self.checkpoint_dir, self.config_name)
        return {
            "model": f"{base}_model.pt",
            "metrics": f"{base}_metrics.json",
            "ewc": f"{base}_ewc.pt",
            "meta": f"{base}_meta.json"
        }

            model.load_state_dict(torch.load(paths["model"], map_location=self.device))
            metrics_engine.load_state(paths["metrics"])
            
            print(f"  [Self-Healing] Resuming from Task {last_task + 2} (Last completed: {last_task+1})")
            return last_task + 1
        return 0
    
    def save_checkpoint(self, t_idx, model, metrics_engine):
        paths = self.get_checkpoint_paths()
        torch.save(model.state_dict(), paths["model"])
        metrics_engine.save_state(paths["metrics"])
        
        with open(paths["meta"], 'w') as f:
            json.dump({"last_completed_task": t_idx}, f)
        print(f"  [Self-Healing] Checkpoint saved for Task {t_idx+1}")

    def load_checkpoint(self, model, metrics_engine):
        paths = self.get_checkpoint_paths()
        if os.path.exists(paths["meta"]):
            with open(paths["meta"], 'r') as f:
                meta = json.load(f)
            last_task = meta["last_completed_task"]

    def train_one_task(self, t_idx, model, train_loader, val_loader):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = copy.deepcopy(model.state_dict())
        start_cls, end_cls = t_idx * 10, (t_idx + 1) * 10

        print(f"  Training for Task {t_idx+1} | Epochs: {self.epochs}")
        for epoch in range(self.epochs):
            model.train()
            # Use progress bar logic if useful, else keep it simple
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # [PHASE IV HARDENING] Use framework train_step instead of manual logic
                # This activates OGD, Consciousness, and Autonomic Health.
                model.train_step(
                    x, 
                    target_data=y, 
                    start_cls=start_cls, 
                    end_cls=end_cls
                )

            val_loss = self.validate(t_idx, model, val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= self.patience: break
        model.load_state_dict(best_model_state)

    def validate(self, t_idx, model, loader):
        model.eval()
        total_loss, start_cls, end_cls = 0, t_idx * 10, (t_idx + 1) * 10
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                total_loss += F.cross_entropy(logits[:, start_cls:end_cls], y - start_cls).item()
        return total_loss / len(loader)

    def run_experiment(self, config: AdaptiveFrameworkConfig):
        print(f"\n[PHASE IV] STARTING: {self.config_name}")
        metrics_engine = MetricsEngine(num_tasks=10, classes_per_task=10, config_name=self.config_name)
        backbone = self.model_factory().to(self.device)
        agent = AdaptiveFramework(backbone, config=config)
        
        # [PHASE IV HARDENING] Agent handles its own optimization internally.
        # We no longer pass an external optimizer or EWC module.

        start_task = self.load_checkpoint(agent, metrics_engine)
        
        for t_idx in range(start_task, 10):
            print(f"--- Task {t_idx+1}/10 ---")
            train_loader, val_loader, _ = self.curriculum.get_task(t_idx)
            self.train_one_task(t_idx, agent, train_loader, val_loader)
            
            # Evaluate all tasks after training
            for eval_idx in range(10):
                _, _, test_loader = self.curriculum.get_task(eval_idx)
                mode = 'task-il' if eval_idx > t_idx else 'class-il'
                acc = self.evaluate(agent, test_loader, t_idx if mode == 'class-il' else eval_idx, mode=mode)
                metrics_engine.update_result(t_idx, eval_idx, acc)
            
            self.save_checkpoint(t_idx, agent, metrics_engine)

        metrics_engine.generate_report()
        metrics_engine.plot_heatmap(filename=os.path.join(ROOT_DIR, 'Phase_IV_Ablation', 'results', f"final_{self.config_name}.png"))

    def evaluate(self, model, loader, target_task_idx, mode='class-il'):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                if mode == 'task-il':
                    start, end = target_task_idx * 10, (target_task_idx + 1) * 10
                    preds = torch.argmax(logits[:, start:end], dim=1) + start
                else:
                    active_classes = (target_task_idx + 1) * 10
                    preds = torch.argmax(logits[:, :active_classes], dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total
