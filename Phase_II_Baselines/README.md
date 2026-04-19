# Phase II: Adversarial Baselines (EWC & A-GEM)

To validate ANTARA, we compare it against the most mathematically established safeguards in Continual Learning research.

## 📐 Implemented Baselines

### 1. Online EWC (Elastic Weight Consolidation)
Implemented according to **Schwarz et al. (2018)**. 
- Unlike standard EWC which anchors to a single past task, **Online EWC** maintains a consolidated Fisher Information Matrix and a running mean of important weights $(\mu)$. 
- This prevents "anchor drift" and allows the model to scale to long task sequences.

### 2. A-GEM (Averaged Gradient Episodic Memory)
Implemented according to **Chaudhry et al. (2019)**.
- **Gradient Projection**: Gradients are projected onto the null space of previous task gradients stored in a Reservoir Buffer.
- **Surgical Orchestration**: Our implementation uses a 5-step grad-injection pattern that prevents primary gradient annihilation during reference backpasses.

## 📂 Key Files
- `baselines.py`: The core wrapper class for EWC anchors and A-GEM projection logic.

## 🚀 Usage (Standalone)
You can verify baseline initialization and mathematical validity by running:
```bash
python Phase_II_Baselines/baselines.py
```
