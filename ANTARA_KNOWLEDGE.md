# ANTARA Knowledge Base
## Adaptive Neural Thinking Architecture for Recursive Autonomy

ANTARA is a "Neuro-Dynamic Wrapper" designed to grant biological-like properties—such as memory, sleep, and homeostasis—to static mathematical models. This knowledge base synthesizes information from the [Official Documentation](http://docs.airbornehrs.in) and the [NeurIPS V2 Paper](file:///c:/Users/surya/.inUse/ultorg/airborne-code/Antara/NeurIPS/V2.pdf).

---

## 1. Core Philosophy: The 'Living AI' Manifesto
Current AI is "Dead on Arrival"—it stops learning after training. ANTARA introduces a persistent efficiency layer that maintains model vitality through:
- **Memory**: Short-term and Long-term Potentiation.
- **Sleep**: Replaying past experiences ("dreaming") using Reservoir Sampling.
- **Pain**: Gradient instability and catastrophic forgetting markers that trigger autonomic repairs.

---

## 2. Architecture: Dual-Cameral Processing
ANTARA V2.0 utilizes a bi-cameral loop that separates reflexive speed from deliberative thought.

### System 1: Reflexive Intelligence (H-MoE)
Handles routine, high-confidence inputs with minimal latency.
- **Mechanism**: Hierarchical Mixture of Experts (H-MoE).
- **Routing**: A Domain Router selects a specialized cluster, followed by an Expert Router selecting specific weights.

### System 2: Deliberative Intelligence (RGW)
Intercepts ambiguous or high-uncertainty inputs for recursive refinement.
- **Mechanism**: Recursive Global Workspace (RGW).
- **Formulation**: $S_{t+1} = FFN(SelfAttn(LN(S_t + MHA(Q=S_t, K=X, V=X))))$
- **Dynamic Recursion**: The number of cycles ($k$) depends on output entropy ($\mathcal{H}$):
  - **Reflex**: $k=1$ if $\mathcal{H} < 0.2$
  - **Deliberation**: $k=3$ if $\mathcal{H} > 0.8$

---

## 3. Key Algorithms

### Orthogonal Gradient Descent (OGD)
Prevents catastrophic forgetting by ensuring new updates do not overwrite critical historical knowledge.
- **Math**: $\nabla \theta_{safe} = \nabla \theta - M_{prev} M_{prev}^T \nabla \theta$
- **Effect**: Gradients are projected onto the null space of previous task manifolds.

### Holographic Associative Memory
Optimizes retrieval of latent vectors by organizing them into **Voronoi cells**.
- **Efficiency**: Reduces retrieval complexity from $O(N)$ to $O(K)$ using centroid-based clustering.

### Learned Optimizer (REINFORCE)
Treats parameter modulation as a Reinforcement Learning problem.
- **Policy**: An LSTM observes $[Loss_t, GradNorm_t, Entropy_t]$.
- **Action**: Outputs a dynamic scalar $\lambda_t \in [0.5, 2.0]$ to adjust the learning rate.

---

## 4. API Reference

### `AdaptiveFramework(model, config=None)`
The primary wrapper class for PyTorch models.

| Method | Description |
| :--- | :--- |
| `train_step(x, target)` | Executes a training step with entropy-based recursion and OGD protection. |
| `register_importance(data_loader)` | Uses Fisher Information to "lock" ancestral knowledge before a new task. |
| `feel()` | Returns the current "Neuromodulatory State" (e.g., CONFIDENT, EXPLORATORY). |

---

## 5. Technical Parameters & Thresholds

> [!TIP]
> Use these thresholds in your `config.yaml` to tune model stability.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Reflex Boundary** | $\mathcal{H} < 0.2$ | Inputs below this entropy skip System 2. |
| **Deliberation Boundary** | $\mathcal{H} > 0.8$ | Triggers full recursive workspace cycles. |
| **Dead Neuron Floor** | $1e-8$ | Gradients below this trigger Kaiming re-initialization. |
| **Seizure Ceiling** | $1e2$ | GradNorms above this trigger automatic clipping/scaling. |
| **Memory Plateau** | $1.5$ | Constant $\mathcal{M}_{plateau}$ for importance saturation. |

---

## 6. Implementation Strategy: Autonomic Repair
ANTARA monitors for two critical failure states:
1. **Comas (Vanishing Weights)**: Detected via low gradient mean; repaired via re-initialization.
2. **Seizures (Exploding Gradients)**: Detected via extreme GradNorm; repaired via gradient scaling and homeostasis restoration.

---

## 7. Basic Usage Example
```python
import torch
from airborne_antara import AdaptiveFramework

# 1. Initialize base model
model = torch.nn.Linear(10, 2)

# 2. Grant Autonomy
agent = AdaptiveFramework(model)

# 3. Normal Training Loop (Now Protected)
x, y = torch.randn(5, 10), torch.randn(5, 2)
agent.train_step(x, target_data=y)

# 4. View Internal State
print(f"Current State: {agent.feel()}")
```
