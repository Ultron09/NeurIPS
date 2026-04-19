# ANTARA NeurIPS Evaluation Suite: The Architectural Autopsy

This repository contains the rigorous, mathematically hardened evaluation suite for the **ANTARA (Adaptive Neural Thinking Architecture for Recursive Analysis)** framework. It is designed to meet the extreme empirical standards of NeurIPS peer review.

## 🔬 Overview
The suite implements a 4-phase gauntlet to validate ANTARA's stability against catastrophic forgetting across non-stationary distributions (Split CIFAR-100). It features **Online EWC (Schwarz et al. 2018)**, **A-GEM (Chaudhry et al. 2019)**, and a **Surgical Logit Masking** engine to prevent softmax fratricide.

## 📂 Project Structure
- **[Phase I: Curriculum](./Phase_I_Curriculum)**: Deterministic 10-task Partitioning of CIFAR-100.
- **[Phase II: Baselines](./Phase_II_Baselines)**: Online EWC and A-GEM reference implementations.
- **[Phase III: Metrics](./Phase_III_Metrics)**: Ri,j Accuracy Matrix, ACC, BWT, and FWT calculation.
- **[Phase IV: Ablation](./Phase_IV_Ablation)**: The parallel, self-healing Gauntlet orchestrator.

## 🚀 Remote Execution Guide

### 1. Environment Setup
Clone the repository and install dependencies on your remote compute node:
```bash
git clone https://github.com/Ultron09/NeurIPS.git
cd NeurIPS
pip install -r requirements.txt
pip install airborne-antara
```

### 2. Parallel Gauntlet Launch
Phase IV is optimized for parallel execution across multiple GPUs. You can launch individual ablation studies simultaneously:

```bash
# Node 1: Full Framework
python Phase_IV_Ablation/vanguard_full.py

# Node 2: Stability Ablation
python Phase_IV_Ablation/ablated_memory.py

# Node 3: Deliberation Ablation
python Phase_IV_Ablation/ablated_consciousness.py

# Node 4: Baseline Control
python Phase_IV_Ablation/naive_control.py
```

### 🛡️ Self-Healing Checkpointing
All Phase IV scripts feature **automatic task-level state recovery**. If a run is interrupted (e.g., spot instance termination), simply re-run the script. It will automatically detect the `results/checkpoints/` directory and resume exactly from the last successfully completed task.

## 📊 Results & Visualization
All metrics and heatmaps are automatically saved to `Phase_IV_Ablation/results/`. These are formatted for high-resolution placement in a LaTeX NeurIPS manuscript.

---
**Lead Researchers:** Suryaansh Prithvijit Singh, Sonya Shelke  
**Organization:** Airborne-Antara Research
