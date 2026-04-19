import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import List, Dict, Optional, Tuple

class EWC(nn.Module):
    """
    Hardened Online EWC for NeurIPS-grade stability.
    Consolidates Fisher Information to prevent anchor drift across many tasks.
    """
    def __init__(self, model: nn.Module, lambda_factor: float = 1000, gamma: float = 1.0):
        super().__init__()
        self.model = model
        self.lambda_factor = lambda_factor
        self.gamma = gamma
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._fisher = {}

    def save_task_weights(self, data_loader, device='cuda'):
        self.model.eval()
        fisher_info = {n: torch.zeros_like(p.data) for n, p in self.params.items()}
        
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            self.model.zero_grad()
            output = self.model(x)
            # Use Log-Softmax for stable Fisher approximation
            loss = F.nll_loss(F.log_softmax(output, dim=1), y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_info[n].data += (p.grad.data ** 2) / len(data_loader)

        for n, f in fisher_info.items():
            if n in self._fisher:
                self._fisher[n] = self.gamma * self._fisher[n] + f
            else:
                self._fisher[n] = f
        
        for n, p in self.model.named_parameters():
            self._means[n] = p.data.clone()

    def penalty(self) -> torch.Tensor:
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self._fisher:
                loss += (self._fisher[n] * (p - self._means[n]) ** 2).sum()
        return loss * (self.lambda_factor / 2)


class ExperienceReplay:
    """Reservoir-sampled Experience Replay for fair ANTARA comparison."""
    def __init__(self, buffer_size: int = 2000):
        self.buffer_size = buffer_size
        self.buffer_x = []
        self.buffer_y = []
        self.seen = 0

    def update(self, x, y):
        for i in range(x.size(0)):
            self.seen += 1
            if len(self.buffer_x) < self.buffer_size:
                self.buffer_x.append(x[i].detach().cpu())
                self.buffer_y.append(y[i].detach().cpu())
            else:
                j = random.randint(0, self.seen - 1)
                if j < self.buffer_size:
                    self.buffer_x[j] = x[i].detach().cpu()
                    self.buffer_y[j] = y[i].detach().cpu()

    def get_batch(self, batch_size=64):
        if not self.buffer_x: return None, None
        idx = np.random.choice(len(self.buffer_x), min(batch_size, len(self.buffer_x)), replace=False)
        return torch.stack([self.buffer_x[i] for i in idx]), torch.as_tensor([self.buffer_y[i] for i in idx])


class AGEM:
    """Correctly-timed A-GEM Gradient Projection."""
    def __init__(self, model, buffer_size=2000):
        self.model = model
        self.buffer = ExperienceReplay(buffer_size)

    def get_ref_grad(self, device='cuda'):
        bx, by = self.buffer.get_batch(64)
        if bx is None: return None
        
        bx, by = bx.to(device), by.to(device)
        self.model.zero_grad()
        loss = F.cross_entropy(self.model(bx), by)
        loss.backward()
        
        grads = []
        for p in self.model.parameters():
            if p.grad is not None: grads.append(p.grad.view(-1))
            else: grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)

    def project(self, curr_grad, ref_grad):
        dot = torch.dot(curr_grad, ref_grad)
        if dot < 0:
            return curr_grad - (dot / (torch.dot(ref_grad, ref_grad) + 1e-9)) * ref_grad
        return curr_grad
