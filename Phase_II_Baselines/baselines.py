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


class DERPlus(nn.Module):
    """
    DER++: Dark Experience Replay (Buzzega et al. 2020).
    The SOTA baseline for replay-based methods.
    Stores x, y, and LOGITS to preserve the functional manifold of prior tasks.
    """
    def __init__(self, model, buffer_size=2000, alpha=0.1, beta=0.5):
        super().__init__()
        self.model = model
        self.buffer_size = buffer_size
        self.alpha = alpha  # Logit matching weight
        self.beta = beta    # Label matching weight
        self.buffer_x = []
        self.buffer_y = []
        self.buffer_logits = []
        self.seen = 0

    def update(self, x, y, logits):
        for i in range(x.size(0)):
            self.seen += 1
            sample_x = x[i].detach().cpu()
            sample_y = y[i].detach().cpu()
            sample_logits = logits[i].detach().cpu()
            
            if len(self.buffer_x) < self.buffer_size:
                self.buffer_x.append(sample_x)
                self.buffer_y.append(sample_y)
                self.buffer_logits.append(sample_logits)
            else:
                j = random.randint(0, self.seen - 1)
                if j < self.buffer_size:
                    self.buffer_x[j] = sample_x
                    self.buffer_y[j] = sample_y
                    self.buffer_logits[j] = sample_logits

    def get_loss(self, x, y, current_logits, device='cuda'):
        if not self.buffer_x: return torch.tensor(0.0).to(device)
        
        # Reservoir batch
        idx = np.random.choice(len(self.buffer_x), min(x.size(0), len(self.buffer_x)), replace=False)
        bx = torch.stack([self.buffer_x[i] for i in idx]).to(device)
        by = torch.as_tensor([self.buffer_y[i] for i in idx]).to(device)
        bl = torch.stack([self.buffer_logits[i] for i in idx]).to(device)
        
        # Logit matching (Dark Experience)
        out_buf = self.model(bx)
        loss_logits = F.mse_loss(out_buf, bl)
        
        # Label matching
        loss_labels = F.cross_entropy(out_buf, by)
        
        return self.alpha * loss_logits + self.beta * loss_labels


class HAT(nn.Module):
    """
    Hard Attention to the Task (Serrà et al. 2018).
    Standard masking baseline for comparative CAS evaluation.
    Utilizes learned task-specific masks to inhibit interference.
    """
    def __init__(self, model, num_tasks=10):
        super().__init__()
        self.model = model
        self.num_tasks = num_tasks
        self.masks = nn.Parameter(torch.ones(num_tasks, 512)) # ResNet18 feature size
        self.gate = nn.Sigmoid()
        # Track cumulative importance to block future updates
        self.register_buffer('cumulative_mask', torch.zeros(512))

    def get_mask(self, t_idx, s=100):
        return self.gate(s * self.masks[t_idx])

    def apply_mask(self, features, t_idx):
        mask = self.get_mask(t_idx)
        return features * mask.view(1, -1, 1, 1) if features.dim() == 4 else features * mask

    def reg_loss(self, t_idx):
        return self.get_mask(t_idx).mean()

    def mask_gradients(self, model):
        """
        [Serrà et al. 2018] Zero gradients for features used by PREVIOUS tasks.
        """
        if self.cumulative_mask.sum() == 0: return
        
        with torch.no_grad():
            # For ResNet18, we target the 'fc' layer which uses the masked features
            if hasattr(model, 'fc'):
                # cumulative_mask is 512-dim, fc.weight is [100, 512]
                if model.fc.weight.grad is not None:
                    # Expand mask to [100, 512] and zero out stale weights
                    m = self.cumulative_mask.view(1, -1).expand_as(model.fc.weight)
                    model.fc.weight.grad.data.mul_(1 - m)

    def update_cumulative_mask(self, t_idx):
        with torch.no_grad():
            self.cumulative_mask = torch.max(self.cumulative_mask, self.get_mask(t_idx).detach())


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
