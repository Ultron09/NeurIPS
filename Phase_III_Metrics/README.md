# Phase III: The Holy Trinity of Metrics

Evaluation in the Continual Learning community requires more than simple loss curves. We utilize the $R_{i,j}$ matrix—where $R_{i,j}$ is the test classification accuracy on task $j$ after observing the last sample from task $i$—to derive the following key metrics.

## Key Metrics

### 1. Average Accuracy (ACC)
- **Definition**: The mean accuracy across all tasks after the final task in the sequence has been learned.
- **Goal**: High overall performance indicates the system's ability to maintain high-capacity storage of multiple tasks.

### 2. Backward Transfer (BWT)
- **Definition**: Measures how learning new tasks affects performance on previously learned tasks.
- **Significance**: ANTARA has demonstrated a BWT of **-0.0491**, indicating near-zero degradation. Replicating this on the Split CIFAR-100 benchmark is a core objective.
- **Equation**: $\frac{1}{T-1} \sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})$

### 3. Forward Transfer (FWT)
- **Definition**: Measures the capability of the model to leverage previous knowledge to perform on a new task without direct training (Zero-Shot capability).
- **Equation**: $\frac{1}{T-1} \sum_{i=2}^T (R_{i-1,i} - \tilde{b}_i)$

## Usage
The `metrics_engine.py` provides a standardized implementation of these equations and generates a **R-Matrix Heatmap** for visual validation.
```python
from metrics_engine import MetricsEngine
engine = MetricsEngine(num_tasks=10)
# Update results during training
engine.update_result(current_task=9, eval_task=0, accuracy=0.85)
# Print report
engine.generate_report()
```
