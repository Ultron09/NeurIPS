import copy
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
def train_single_task(model, train_loader, val_loader, optimizer, t_idx, device='cuda', 
                      ewc_module=None, agem_module=None, replay_buffer=None,
                      epochs=10, patience=3):
    """
    Unified training logic with Global Logit Calibration and Surgical Gradient Injection.
    """
    best_loss = float('inf')
    best_model = copy.deepcopy(model.state_dict())
    no_improve = 0
    total_step_time = 0.0
    total_steps = 0
    
    # Global Output Boundaries
    seen_classes = (t_idx + 1) * 10
    
    for epoch in range(epochs):
        model.train()
        train_iter = tqdm(train_loader, desc=f"Task {t_idx} Epoch {epoch}")
        for x, y in train_iter:
            start_time = time.time()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # --- AGEM REFERENCE GRADIENT ---
            ref_grad = None
            if agem_module:
                ref_grad = agem_module.get_ref_grad(device=device)
                optimizer.zero_grad() # Clear ref_grad influence from main optimizer
            
            # --- MAIN FORWARD PASS (Global Header for calibration) ---
            logits = model(x)
            # Use all seen classes in the denominator to keep logits calibrated
            loss = F.cross_entropy(logits[:, :seen_classes], y)
            
            # --- REPLAY INJECTION ---
            if replay_buffer:
                rx, ry = replay_buffer.get_batch(32)
                if rx is not None:
                    r_logits = model(rx.to(device))
                    loss += F.cross_entropy(r_logits[:, :seen_classes], ry.to(device))
            
            # --- EWC PENALTY ---
            if ewc_module:
                loss += ewc_module.penalty()
                
            loss.backward()
            
            # --- AGEM PROJECTION ---
            if agem_module and ref_grad is not None:
                curr_grad = []
                for p in model.parameters():
                    if p.grad is not None: curr_grad.append(p.grad.view(-1))
                    else: curr_grad.append(torch.zeros(p.numel(), device=device))
                curr_grad = torch.cat(curr_grad)
                
                projected = agem_module.project(curr_grad, ref_grad)
                
                # Manual Gradient Injection
                pointer = 0
                for p in model.parameters():
                    p.grad = projected[pointer:pointer+p.numel()].view_as(p).clone()
                    pointer += p.numel()

            optimizer.step()
            
            # --- POST-OPTIMIZATION BUFFER STORAGE (Fixing pollution) ---
            if agem_module: agem_module.buffer.update(x, y)
            if replay_buffer: replay_buffer.update(x, y)
            
            total_step_time += (time.time() - start_time) * 1000 # to ms
            total_steps += 1

        # Validation with early stopping
        val_loss = validate(model, val_loader, seen_classes, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience: break
            
    model.load_state_dict(best_model)
    avg_step_time = total_step_time / total_steps if total_steps > 0 else 0
    return avg_step_time

def validate(model, loader, seen_classes, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits[:, :seen_classes], y).item()
    return total_loss / len(loader)
