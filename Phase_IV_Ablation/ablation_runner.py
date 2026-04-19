import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Robust Path Resolution: Ensure sister directories are in PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub_dir in ['Phase_I_Curriculum', 'Phase_II_Baselines', 'Phase_III_Metrics']:
    path = os.path.join(ROOT_DIR, sub_dir)
    if path not in sys.path:
        sys.path.append(path)

from curriculum import SplitCIFAR100, set_seed
from metrics_engine import MetricsEngine
from baselines import EWC, AGEM, ExperienceReplay
import argparse
import copy
from typing import Optional

class AblationOrchestrator:
    """
    Standardizes the three runs for the Architectural Autopsy with Surgical Rigor.
    Resolves FWT Paradox and Softmax Fratricide.
    """
    def __init__(self, model_factory, epochs=50, patience=5, seed=42, device='cuda'):
        self.model_factory = model_factory
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = device
        
        set_seed(self.seed)
        self.curriculum = SplitCIFAR100(seed=self.seed)

    def train_one_task(self, t_idx, model, train_loader, val_loader, optimizer, 
                       ewc_module: Optional[EWC] = None, 
                       agem_module: Optional[AGEM] = None):
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = copy.deepcopy(model.state_dict())
        
        # Task Boundary (for Surgical Loss Isolation)
        start_cls = t_idx * 10
        end_cls = (t_idx + 1) * 10

        print(f"  Training for Task {t_idx+1} | Isolation: [{start_cls}-{end_cls-1}]")

        for epoch in range(self.epochs):
            model.train()
            total_train_loss = 0
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                
                logits = model(x)
                
                # SURGICAL LOSS: Isolate to the current task's output head
                # This prevents "Softmax Fratricide" (destruction of past/future heads)
                task_logits = logits[:, start_cls:end_cls]
                task_y = y - start_cls # Map absolute labels (e.g. 20-29) to local head (0-9)
                
                loss = F.cross_entropy(task_logits, task_y)
                
                if ewc_module:
                    loss += ewc_module.penalty()
                
                loss.backward()
                
                if agem_module:
                    agem_module.store_samples(x, y)
                    g_current = agem_module.get_grad_vector(model)
                    g_ref = agem_module.get_ref_gradient(device=self.device)
                    if g_ref is not None:
                        g_proj = agem_module.project_gradient(g_current, g_ref)
                        agem_module.inject_grad_vector(model, g_proj)
                
                optimizer.step()
                total_train_loss += loss.item()

            # Validation with the same isolation logic
            val_loss = self.validate(t_idx, model, val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            
            if epoch % 5 == 0 or epochs_no_improve == 0:
                print(f"    Epoch {epoch:02d} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

            if epochs_no_improve >= self.patience:
                print(f"    [Convergence] Early stopping at epoch {epoch}")
                break
        
        model.load_state_dict(best_model_state)

    def validate(self, t_idx, model, loader):
        model.eval()
        total_loss = 0
        start_cls = t_idx * 10
        end_cls = (t_idx + 1) * 10
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                total_loss += F.cross_entropy(logits[:, start_cls:end_cls], y - start_cls).item()
        return total_loss / len(loader)

    def run_experiment(self, name: str, antara_config: dict, use_ewc=False, use_agem=False):
        print(f"\n[PHASE IV] RUNNING ABLATION: {name}")
        metrics_engine = MetricsEngine(num_tasks=10, classes_per_task=10)
        
        backbone = self.model_factory().to(self.device)
        agent = AdaptiveFramework(backbone, config=antara_config)
        optimizer = torch.optim.SGD(agent.parameters(), lr=0.01, momentum=0.0)
        
        ewc = EWC(agent, gamma=1.0) if use_ewc else None
        agem = AGEM(agent) if use_agem else None
        
        for t_idx in range(10):
            print(f"--- Task {t_idx+1}/10 ---")
            train_loader, val_loader, _ = self.curriculum.get_task(t_idx)
            
            self.train_one_task(t_idx, agent, train_loader, val_loader, optimizer, ewc, agem)
            
            if ewc:
                ewc.save_task_weights(train_loader)
            
            # Complex Evaluation Logic
            for eval_idx in range(10):
                _, _, test_loader = self.curriculum.get_task(eval_idx)
                
                # Zero-Shot (Forward Transfer) must use isolated heads (Task-IL)
                # Global Accuracy (ACC/BWT) must use all seen heads (Class-IL)
                if eval_idx > t_idx:
                    # Forward Transfer Mode
                    acc = self.evaluate(agent, test_loader, eval_idx, mode='task-il')
                else:
                    # Class-Incremental Mode (Acc/BWT)
                    acc = self.evaluate(agent, test_loader, t_idx, mode='class-il')
                
                metrics_engine.update_result(t_idx, eval_idx, acc)
        
        metrics_engine.generate_report()
        metrics_engine.plot_heatmap(filename=f"ablation_{name.lower().replace(' ', '_')}.png")

    def evaluate(self, model, loader, target_task_idx, mode='class-il'):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                
                if mode == 'task-il':
                    # FORWARD TRANSFER: Isolate to the specific target task's head
                    start = target_task_idx * 10
                    end = (target_task_idx + 1) * 10
                    preds = torch.argmax(logits[:, start:end], dim=1) + start
                else:
                    # CLASS-IL (ACC/BWT): Pick from all classes seen so far
                    active_classes = (target_task_idx + 1) * 10
                    preds = torch.argmax(logits[:, :active_classes], dim=1)
                
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Smoking Test: Ablation Runner...")
    def tiny_factory():
        return nn.Sequential(nn.Flatten(), nn.Linear(3072, 100))
    
    # Verify Orchestrator can be instantiated (Proof of import resolution)
    orchestrator = AblationOrchestrator(model_factory=tiny_factory, epochs=1, patience=1)
    print("Orchestrator Initialized. Path Resolution Verified.")
    print("Verification Successful.")
