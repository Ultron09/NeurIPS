import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
from pathlib import Path
import os
import urllib.request
import zipfile

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
        
        # [V29] Training augmentation: Standard CL benchmark transforms
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        # Test/val: No augmentation
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        self.train_set = datasets.CIFAR100(root=root, train=True, download=True, transform=self.train_transform)
        self.test_set = datasets.CIFAR100(root=root, train=False, download=True, transform=self.test_transform)
        
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


class SplitMNIST:
    """Standard 5-task MNIST split (2 classes per task)."""
    def __init__(self, root='./data', seed=42, batch_size=64, pin_memory=False):
        self.root = root
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        set_seed(seed)
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Convert 1-ch to 3-ch
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
        ])
        
        self.train_set = datasets.MNIST(root=root, train=True, download=True, transform=self.transform)
        self.test_set = datasets.MNIST(root=root, train=False, download=True, transform=self.transform)
        self.task_classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def get_task(self, task_id):
        if task_id < 0 or task_id >= 5: raise ValueError("Task ID 0-4")
        target_classes = self.task_classes[task_id]
        train_idx = [i for i, l in enumerate(self.train_set.targets) if l in target_classes]
        test_idx = [i for i, l in enumerate(self.test_set.targets) if l in target_classes]
        
        split = int(0.9 * len(train_idx))
        t_idx = train_idx[:split]
        v_idx = train_idx[split:]
        
        return DataLoader(Subset(self.train_set, t_idx), batch_size=self.batch_size, shuffle=True), \
               DataLoader(Subset(self.train_set, v_idx), batch_size=self.batch_size, shuffle=False), \
               DataLoader(Subset(self.test_set, test_idx), batch_size=self.batch_size, shuffle=False)


class SplitTinyImageNet:
    """
    Split-TinyImageNet-200 Curriculum.
    200 classes partitioned into 10 sequential tasks (20 classes each).
    Expects data at {root}/tiny-imagenet-200/
    """
    def __init__(self, root='./data', seed=42, batch_size=64, pin_memory=False):
        self.root = root
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        set_seed(seed)
        
        # [V26] NATIVE RESOLUTION: Use 64x64 for TinyImageNet (NeurIPS Killshot)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # TinyImageNet usually requires ImageFolder structure
        # Expected structure: root/tiny-imagenet-200/train/n01234567/images/
        self.root_path = Path(root)
        self.tiny_dir = self.root_path / "tiny-imagenet-200"
        
        if not self.tiny_dir.exists():
            self._download_and_extract()
            
        train_dir = self.tiny_dir / "train"
        test_dir = self.tiny_dir / "val" # Using val as test for CL
        
        if not train_dir.exists():
            raise FileNotFoundError(f"TinyImageNet train directory not found at {train_dir} even after download attempt.")
            
        self.train_set = datasets.ImageFolder(root=str(train_dir), transform=self.transform)
        self.test_set = datasets.ImageFolder(root=str(test_dir), transform=self.transform)
        
        # Map classes to 0-199
        self.classes = list(range(200))
        random.shuffle(self.classes)
        
        self.task_classes = [list(range(i, i+20)) for i in range(0, 200, 20)]

    def _download_and_extract(self):
        """Downloads, extracts, and REFORMATS TinyImageNet-200."""
        print(f"  [SYSTEM] TinyImageNet not found in {self.tiny_dir}")
        print(f"  [SYSTEM] Attempting download from Stanford mirrors...")
        self.root_path.mkdir(parents=True, exist_ok=True)
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = self.root_path / "tiny-imagenet-200.zip"
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"  [SYSTEM] Download complete. Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_path)
            
            # [REFORMATTING LOGIC] Move val images to class folders (The PyPI Fix)
            val_dir = self.tiny_dir / "val"
            val_images_dir = val_dir / "images"
            val_annotations = val_dir / "val_annotations.txt"
            
            print(f"  [SYSTEM] Reformatting validation set for ImageFolder compatibility...")
            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name, class_id = parts[0], parts[1]
                    target_dir = val_dir / class_id / "images"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    os.rename(val_images_dir / img_name, target_dir / img_name)
            
            if val_images_dir.exists(): os.rmdir(val_images_dir)
            if zip_path.exists(): os.remove(zip_path)
            print(f"  [SYSTEM] TinyImageNet ready and reformatted.")
        except Exception as e:
            print(f"  [CRITICAL] Data Setup Failed: {e}")
            raise e

    def get_task(self, task_id):
        if self.train_set is None:
            raise FileNotFoundError("TinyImageNet dataset files missing. Please download to ./data/tiny-imagenet-200/")
            
        if task_id < 0 or task_id >= 10:
            raise ValueError("Task ID must be between 0 and 9")
            
        target_classes = self.task_classes[task_id]
        
        # ImageFolder targets are class indices [0-199] based on folder order
        train_indices = [i for i, label in enumerate(self.train_set.targets) if label in target_classes]
        test_indices = [i for i, label in enumerate(self.test_set.targets) if label in target_classes]
        
        split = int(0.9 * len(train_indices))
        t_idx = train_indices[:split]
        v_idx = train_indices[split:]
        
        return DataLoader(Subset(self.train_set, t_idx), batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory), \
               DataLoader(Subset(self.train_set, v_idx), batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory), \
               DataLoader(Subset(self.test_set, test_indices), batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory)
