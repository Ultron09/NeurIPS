import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

def set_seed(seed=42):
    """Enforces bit-for-bit deterministic reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[REPRODUCIBILITY] Global seed set to {seed}")

class SplitCIFAR100:
    """
    Standardizes the CIFAR-100 partition into 10 sequential tasks.
    Ensures deterministic data splits and reproducible curriculum.
    """
    def __init__(self, root='./data', batch_size=64, val_split=0.1, seed=42):
        self.root = root
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        
        # Standard Normalization for CIFAR
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        # Load datasets
        self.train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=self.transform)
        self.test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=self.transform)
        
        # Deterministic class ordering
        self.class_order = list(range(100))
        self.tasks = [self.class_order[i:i+10] for i in range(0, 100, 10)]

    def get_task(self, task_id):
        """Returns DataLoaders for (train, val, test) for a specific task."""
        if task_id < 0 or task_id >= 10:
            raise ValueError("Task ID must be between 0 and 9")
            
        target_classes = self.tasks[task_id]
        
        # Filter indices
        task_train_indices = [i for i, label in enumerate(self.train_set.targets) if label in target_classes]
        test_indices = [i for i, label in enumerate(self.test_set.targets) if label in target_classes]
        
        # Deterministic Split using fixed seed generator
        g = torch.Generator()
        g.manual_seed(self.seed + task_id) # Task-aware but deterministic split
        
        total_train = len(task_train_indices)
        val_size = int(total_train * self.val_split)
        train_size = total_train - val_size
        
        indices = torch.tensor(task_train_indices)
        perm = torch.randperm(total_train, generator=g)
        
        train_idx = indices[perm[:train_size]]
        val_idx = indices[perm[train_size:]]
        
        # Ensure DataLoaders also use the generator for shuffle
        train_loader = DataLoader(Subset(self.train_set, train_idx.tolist()), 
                                  batch_size=self.batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(Subset(self.train_set, val_idx.tolist()), 
                                batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(Subset(self.test_set, test_indices), 
                                 batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
