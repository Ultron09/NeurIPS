# ANTARA NeurIPS Evaluation Suite: The Strategic Gauntlet

This repository contains the rigorous, mathematically hardened evaluation suite for the **ANTARA (Adaptive Neural Thinking Architecture for Recursive Analysis)** framework. It is designed to meet the extreme empirical standards of NeurIPS peer review.

## 🔬 Scientific Objective
The suite executes an **Architectural Autopsy** to validate the core hypotheses of the ANTARA framework:
1.  **System-Level Synergy**: Prove that intelligence and stability emerge from the interaction of System 1 (OGD) and System 2 (RGW).
2.  **Efficiency Superiority**: Demonstrate that ANTARA achieves SOTA retention (BWT > -0.05) with strictly lower compute overhead than projection-based methods like A-GEM.
3.  **Strict Data Isolation**: Prove stability without the "crutch" of raw data rehearsal.

## 📂 Project Structure
- **[Phase I: Curriculum](./Phase_I_Curriculum)**: Deterministic 10-task Partitioning of CIFAR-100.
- **[Phase II: Baselines](./Phase_II_Baselines)**: Standardized EWC, ER, and A-GEM implementations.
- **[Phase III: Metrics](./Phase_III_Metrics)**: Strategic telemetry tracking (ACC, BWT, FWT, Step Latency, Memory).
- **[Phase IV: Ablation](./Phase_IV_Ablation)**: Unified branch runner for the 7-branch gauntlet.

## 🚀 Execution Guide (Remote Parity)

### 1. Parallel Task Launch
Phase IV uses a unified runner to ensure exact hyperparameter parity across all branches. Run these in parallel on your compute clusters:

```bash
# Node 1: Full Framework
python Phase_IV_Ablation/benchmark_runner.py --method ANTARA_FULL

# Node 2: Synergy Ablations
python Phase_IV_Ablation/benchmark_runner.py --method ANTARA_RGW_ONLY
python Phase_IV_Ablation/benchmark_runner.py --method ANTARA_OGD_ONLY

# Node 3: Competitive Baselines
python Phase_IV_Ablation/benchmark_runner.py --method AGEM
python Phase_IV_Ablation/benchmark_runner.py --method EWC
python Phase_IV_Ablation/benchmark_runner.py --method REPLAY
```

### 2. Distributed Team Workflow (4 Laptops)
To hit the 10-day deadline, assign the 7 branches to your team members as follows:

- **Laptop 1 (Lead)**: `ANTARA_FULL`
- **Laptop 2**: `A-GEM`, `NAIVE`
- **Laptop 3**: `EWC`, `REPLAY`
- **Laptop 4**: `ANTARA_RGW_ONLY`, `ANTARA_OGD_ONLY`

#### Parity Protocol:
1. Every team member must run: `pip install -r requirements.txt`
2. Run assigned commands: `python Phase_IV_Ablation/benchmark_runner.py --method <ASSIGNED_METHOD>`

#### 🔴 Live Monitoring (Weights & Biases):
To see the "Live Thing" on your Lead laptop, each team member should run:
```bash
pip install wandb
wandb login
python Phase_IV_Ablation/benchmark_runner.py --method <ASSIGNED_METHOD> --wandb
```
Open your [W&B Dashboard](https://wandb.ai/home) to see the 4-laptop cluster results updating in real-time.

#### Results Synchronization:
Once runs are complete, send all `.json` files from the `results/` folder to the Lead laptop. Place them in the root `results/` directory and run:
```bash
python Phase_IV_Ablation/aggregate_results.py
```
This generates the final `results/final_strategic_report.md`.

### 📊 Strategic Results Aggregation
After the branches complete, generate the unified comparison report:
```bash
python Phase_IV_Ablation/aggregate_results.py
```
This generates `results/final_strategic_report.md`, which contains the publication-ready comparison table.

---
**Lead Researchers:** Suryaansh Prithvijit Singh, Sonya Shelke  
**Organization:** Airborne-Antara Research
