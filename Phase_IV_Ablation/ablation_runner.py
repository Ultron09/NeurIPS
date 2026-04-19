import torch
import torch.nn as nn
import torch.nn.functional as F
from airborne_antara import AdaptiveFramework
from curriculum import SplitCIFAR100, set_seed
from metrics_engine import MetricsEngine
from baselines import EWC, AGEM, ExperienceReplay
import argparse
import copy
from typing import Optional

class AblationOrchestrator:
    """
    Standardizes the three runs for the Architectural Autopsy with Surgical Rigor.
    Ensures Deterministic REPRODUCIBILITY, SGD Optimizer consistency, and Logit MASKING.
    """
    def __init__(self, model_factory, epochs=50, patience=5, seed=42, device='cuda'):
        self.model_factory = model_factory
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = device
        
        # Enforce reproducibility at start
        set_seed(self.seed)
        self.curriculum = SplitCIFAR100(seed=self.seed)

    def train_one_task(self, t_idx, model, train_loader, val_loader, optimizer, 
                       ewc_module: Optional[EWC] = None, 
                       agem_module: Optional[AGEM] = None):
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = copy.deepcopy(model.state_dict())
        
        # Determine the Active Output Space (Class-Incremental Masking)
        active_classes = (t_idx + 1) * 10

        print(f"  Training for Task {t_idx+1} | up to {self.epochs} epochs | Masking: [0-{active_classes-1}]")

        for epoch in range(self.epochs):
            model.train()
            total_train_loss = 0
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                
                # Forward Pass with Masking (Prevent Softmax Fratricide)
                logits = model(x)
                masked_logits = logits[:, :active_classes]
                
                # CrossEntropy on active slice
                loss = F.cross_entropy(masked_logits, y)
                
                # EWC Penalty
                if ewc_module:
                    loss += ewc_module.penalty()
                
                # 1. Backward Pass
                loss.backward()
                
                # 2. A-GEM Surgical Gradient Orchestration (SGD compatible)
                if agem_module:
                    agem_module.store_samples(x, y)
                    g_current = agem_module.get_grad_vector(model)
                    g_ref = agem_module.get_ref_gradient(device=self.device)
                    if g_ref is not None:
                        g_proj = agem_module.project_gradient(g_current, g_ref)
                        agem_module.inject_grad_vector(model, g_proj)
                
                optimizer.step()
                total_train_loss += loss.item()

            # Validation Pass (with Masking)
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
        active_classes = (t_idx + 1) * 10
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                total_loss += F.cross_entropy(logits[:, :active_classes], y).item()
        return total_loss / len(loader)

    def run_experiment(self, name: str, antara_config: dict, use_ewc=False, use_agem=False):
        print(f"\n[PHASE IV] RUNNING ABLATION: {name}")
        metrics_engine = MetricsEngine(num_tasks=10, classes_per_task=10)
        
        # Fresh backbone and SGD with ZERO momentum
        # This prevents Adam state from pull-back against A-GEM projections
        backbone = self.model_factory().to(self.device)
        agent = AdaptiveFramework(backbone, config=antara_config)
        optimizer = torch.optim.SGD(agent.parameters(), lr=0.01, momentum=0.0)
        
        ewc = EWC(agent, gamma=1.0) if use_ewc else None
        agem = AGEM(agent) if use_agem else None
        
        for t_idx in range(10):
            print(f"--- Task {t_idx+1}/10 ---")
            train_loader, val_loader, _ = self.curriculum.get_task(t_idx)
            
            self.train_one_task(t_idx, agent, train_loader, val_loader, optimizer, ewc, agem)
            
            # EWC post-task anchor update
            if ewc:
                ewc.save_task_weights(train_loader)
            
            # Global Evaluation (across all seen/unseen tasks)
            for eval_idx in range(10):
                # NOTE: Evaluation only masks classes up to the highest task seen so far (optional)
                # Standard CL literature evaluates on ALL classes at end of each task
                _, _, test_loader = self.curriculum.get_task(eval_idx)
                acc = self.evaluate(agent, test_loader, t_idx)
                metrics_engine.update_result(t_idx, eval_idx, acc)
        
        metrics_engine.generate_report()
        metrics_engine.plot_heatmap(filename=f"ablation_{name.lower().replace(' ', '_')}.png")

    def evaluate(self, model, loader, current_max_task):
        model.eval()
        correct = 0
        total = 0
        # For fair evaluation in class-incremental learning, we mask to the classes seen so far
        # This prevents predicting classes that have not been encountered.
        active_classes = (current_max_task + 1) * 10
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
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
    
    # Ready for Orchestration
