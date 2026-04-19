import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
from typing import List, Dict, Optional, Tuple

class EWC(nn.Module):
    """
    Online Elastic Weight Consolidation (Online EWC).
    Reference: Schwarz et al. "Progress & Compress: A Scalable Framework for Continual Learning"
    Fixes the anchor-drift flaw by consolidating Fisher importance across tasks.
    """
    def __init__(self, model: nn.Module, lambda_factor: float = 400, gamma: float = 1.0):
        super().__init__()
        self.model = model
        self.lambda_factor = lambda_factor
        self.gamma = gamma # Decay factor for Online EWC
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        
        # Consistently maintained across all tasks
        self._means = {}
        self._consolidated_fisher = {}

    def calculate_importance(self, data_loader):
        self.model.eval()
        fisher_info = {}
        for n, p in self.params.items():
            fisher_info[n] = torch.zeros_like(p.data)

        device = next(self.model.parameters()).device
        for input, label in data_loader:
            input, label = input.to(device), label.to(device)
            self.model.zero_grad()
            output = self.model(input)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_info[n].data += p.grad.data ** 2 / len(data_loader)

        return fisher_info

    def save_task_weights(self, data_loader):
        """Updates the running Online EWC anchors."""
        print(f"[ONLINE EWC] Consolidating Fisher info (gamma={self.gamma})...")
        
        # 1. Update Fisher Importance: F_new = gamma * F_old + F_current
        current_fisher = self.calculate_importance(data_loader)
        for n, f in current_fisher.items():
            if n in self._consolidated_fisher:
                self._consolidated_fisher[n] = self.gamma * self._consolidated_fisher[n] + f
            else:
                self._consolidated_fisher[n] = f

        # 2. Update Means: Anchor to the latest converged weights
        for n, p in self.model.named_parameters():
            self._means[n] = p.data.clone()

    def penalty(self) -> torch.Tensor:
        """Computes the scalar Online EWC penalty."""
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self._consolidated_fisher:
                # Penalty = 0.5 * lambda * Consolidated_Fisher * (theta - theta_anchor)^2
                _loss = self._consolidated_fisher[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss * (self.lambda_factor / 2)


class ExperienceReplay:
    """Experience Replay (ER) with algorithmic Reservoir Sampling."""
    def __init__(self, buffer_size: int = 2000):
        self.buffer_size = buffer_size
        self.buffer_x = []
        self.buffer_y = []
        self.total_seen = 0

    def update_buffer(self, x: torch.Tensor, y: torch.Tensor):
        for i in range(x.size(0)):
            self.total_seen += 1
            if len(self.buffer_x) < self.buffer_size:
                self.buffer_x.append(x[i].detach().cpu())
                self.buffer_y.append(y[i].detach().cpu())
            else:
                j = random.randint(0, self.total_seen - 1)
                if j < self.buffer_size:
                    self.buffer_x[j] = x[i].detach().cpu()
                    self.buffer_y[j] = y[i].detach().cpu()

    def get_batch(self, batch_size: int = 64) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.buffer_x:
            return None, None
        indices = np.random.choice(len(self.buffer_x), min(batch_size, len(self.buffer_x)), replace=False)
        batch_x = torch.stack([self.buffer_x[i] for i in indices])
        batch_y = torch.as_tensor([self.buffer_y[i] for i in indices])
        return batch_x, batch_y


class AGEM:
    """Averaged Gradient Episodic Memory (A-GEM) with non-destructive gradient handling."""
    def __init__(self, model: nn.Module, memory_size: int = 2048):
        self.model = model
        self.memory_size = memory_size
        self.memory_x = []
        self.memory_y = []
        self.total_seen = 0

    @staticmethod
    def get_grad_vector(model: nn.Module) -> torch.Tensor:
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)

    @staticmethod
    def inject_grad_vector(model: nn.Module, grad_vector: torch.Tensor):
        pointer = 0
        for p in model.parameters():
            numel = p.numel()
            p.grad = grad_vector[pointer:pointer+numel].view_as(p).clone()
            pointer += numel

    def store_samples(self, x, y):
        for i in range(x.size(0)):
            self.total_seen += 1
            if len(self.memory_x) < self.memory_size:
                self.memory_x.append(x[i].detach().cpu())
                self.memory_y.append(y[i].detach().cpu())
            else:
                j = random.randint(0, self.total_seen - 1)
                if j < self.memory_size:
                    self.memory_x[j] = x[i].detach().cpu()
                    self.memory_y[j] = y[i].detach().cpu()

    def get_ref_gradient(self, device='cpu') -> Optional[torch.Tensor]:
        if not self.memory_x:
            return None
        
        batch_size = 64
        indices = np.random.choice(len(self.memory_x), min(batch_size, len(self.memory_x)), replace=False)
        batch_x = torch.stack([self.memory_x[i] for i in indices]).to(device)
        batch_y = torch.as_tensor([self.memory_y[i] for i in indices]).to(device)
        
        self.model.zero_grad()
        outputs = self.model(batch_x)
        loss = F.cross_entropy(outputs, batch_y)
        loss.backward()
        
        return self.get_grad_vector(self.model)

    def project_gradient(self, current_grad: torch.Tensor, ref_grad: torch.Tensor) -> torch.Tensor:
        dot_product = torch.dot(current_grad, ref_grad)
        if dot_product < 0:
            projection = (dot_product / (torch.dot(ref_grad, ref_grad) + 1e-9)) * ref_grad
            return current_grad - projection
        return current_grad

if __name__ == "__main__":
    print("Testing Baselines...")
    model = nn.Linear(10, 10)
    ewc = EWC(model)
    er = ExperienceReplay(buffer_size=100)
    agem = AGEM(model)
    print("Baselines Initialized Successfully.")
