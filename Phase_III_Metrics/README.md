# Phase III: Metrics Engine (Ri,j Matrix)

The Metrics Engine is the empirical heart of the evaluation suite, responsible for calculating the standardized "Triple Threat" of Continual Learning metrics.

## 📊 Standardized Metrics

### 1. Average Accuracy (ACC)
The mean performance across all seen tasks after the curriculum is completed.

### 2. Backward Transfer (BWT)
The influence of learning new tasks on the performance of previously learned tasks. 
- **Goal**: $> -0.05$ (indicating minimal forgetting).

### 3. Forward Transfer (FWT)
The zero-shot influence of past knowledge on future tasks.
- **Goal**: $> 0.10$ (outperforming the random-guessing baseline).
- **Hardening**: Our engine uses a 10% Task-Isolated baseline for FWT to ensure mathematical validity.

## 🖼️ Accuracy Heatmaps
The Ri,j Accuracy Matrix is visualized as a heatmap, where the row $i$ represents the training phase and column $j$ represents the evaluation domain. 
- **Top-right triangle**: Forward Transfer potential.
- **Lower-left triangle**: Stability/Forgetting signals.

## 🚀 Usage (Standalone)
Verify report generation logic:
```bash
python Phase_III_Metrics/metrics_engine.py
```
