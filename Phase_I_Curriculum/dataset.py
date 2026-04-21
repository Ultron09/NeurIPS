import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SplitCIFAR100:
    """
    Deterministic Split-CIFAR100 Curriculum.
    Partitions 100 classes into 10 sequential tasks (10 classes each).
    """
    def __init__(self, root='./data', seed=42, batch_size=64, pin_memory=False):
        self.root = root
        self.seed = seed
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        set_seed(seed)
        
        # Standard CIFAR-100 Normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        self.train_set = datasets.CIFAR100(root=root, train=True, download=True, transform=self.transform)
        self.test_set = datasets.CIFAR100(root=root, train=False, download=True, transform=self.transform)
        
        # Deterministic class permutation
        self.classes = list(range(100))
        random.shuffle(self.classes)
        
        # Remap targets so they fall continuously from 0 to 99 across tasks
        class_map = {orig: new for new, orig in enumerate(self.classes)}
        self.train_set.targets = [class_map[t] for t in self.train_set.targets]
        self.test_set.targets = [class_map[t] for t in self.test_set.targets]
        
        self.task_classes = [list(range(i, i+10)) for i in range(0, 100, 10)]

    def get_task(self, task_id):
        """Returns loaders for a specific task domain."""
        if task_id < 0 or task_id >= 10:
            raise ValueError("Task ID must be between 0 and 9")
            
        target_classes = self.task_classes[task_id]
        
        # Filter indices
        train_indices = [i for i, label in enumerate(self.train_set.targets) if label in target_classes]
        test_indices = [i for i, label in enumerate(self.test_set.targets) if label in target_classes]
        
        # Split train into train/val (90/10)
        split = int(0.9 * len(train_indices))
        train_idx = train_indices[:split]
        val_idx = train_indices[split:]
        
        train_loader = DataLoader(Subset(self.train_set, train_idx), batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory)
        val_loader = DataLoader(Subset(self.train_set, val_idx), batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory)
        test_loader = DataLoader(Subset(self.test_set, test_indices), batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory)
        
        return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("Initializing Split-CIFAR100 Curriculum...")
    curriculum = SplitCIFAR100()
    print(f"Task Boundaries: {curriculum.task_classes}")
    t1_train, _, _ = curriculum.get_task(0)
    print(f"Task 1 Loaders Ready. Batch size: {next(iter(t1_train))[0].size(0)}")
