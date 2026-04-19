# Phase II: The Adversarial Baselines

To prove the superiority of the ANTARA framework, we evaluate it against the established State-of-the-Art (SOTA) baselines in Continual Learning.

## Integrated Baselines

### 1. Elastic Weight Consolidation (EWC)
- **Concept**: A regularization-based approach that slows down learning on weights important for past tasks.
- **Mechanism**: Utilizes the diagonal of the Fisher Information Matrix (FIM) as a proxy for parameter importance.
- **Weakness**: Often overly rigid, leading to poor plastic performance in high-complexity regimes like Split CIFAR-100.

### 2. Experience Replay (ER)
- **Concept**: The standard memory-buffer baseline.
- **Mechanism**: Interleaves a small subset of stored past samples with the current task's mini-batches.
- **Weakness**: Vulnerable to the "recency bias" and limited by buffer size constraints.

### 3. Averaged Gradient Episodic Memory (A-GEM)
- **Concept**: Gradient projection as a constraint.
- **Mechanism**: Ensures that the gradient update for the current task does not increase the loss on a reference batch from the episodic memory.
- **Relationship to ANTARA**: This is ANTARA's closest philosophical rival, as it also uses gradient projection (similar to OGD). Demonstrating ANTARA's superior performance with lower computational overhead is a key goal.

## Implementation Standard
All baselines are implemented in **native PyTorch** to ensure maximum transparency during the NeurIPS review process.

## Usage
Import these from `baselines.py` within your training loop:
```python
from baselines import EWC, ExperienceReplay, AGEM
```
