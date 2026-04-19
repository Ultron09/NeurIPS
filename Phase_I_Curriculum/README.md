# Phase I: The Split CIFAR-100 Curriculum

This module handles the deterministic partitioning of the CIFAR-100 dataset into a non-stationary stream of 10 tasks.

## 🏗️ The Curriculum Strategy
Standard CIFAR-100 is divided into 10 sequential tasks, with 10 disjoint classes per task. 

- **Task 1**: Classes 0-9
- **Task 2**: Classes 10-19
- ...
- **Task 10**: Classes 90-99

## 🛡️ Reproducibility Rigor
To meet NeurIPS standards for bit-for-bit reproducibility, the `SplitCIFAR100` class utilizes a global `set_seed` utility and task-isolated `torch.Generator` instances. This ensures that the training/validation splits (90/10) and data ordering remain identical across every independent execution of the suite.

## 📂 Key Files
- `curriculum.py`: The core dataset partitioner and seed manager.

## 🚀 Usage (Standalone)
You can verify the curriculum splits by running:
```bash
python Phase_I_Curriculum/curriculum.py
```
This will download CIFAR-100 and print the task boundary indices.
