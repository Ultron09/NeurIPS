# NeurIPS Strategic Gauntlet: Final Comparison Report

## 🏁 Comparative Performance Table

| Method          |   ACC (↑) |   BWT (↑) |   FWT (↑) |   Step Time (ms) |   Memory (MB) |
|:----------------|----------:|----------:|----------:|-----------------:|--------------:|
| ANTARA_FULL     |    0.0122 |   -0.0296 |      -0.1 |          2773.44 |             0 |
| ANTARA_RGW_ONLY |    0.0116 |   -0.0172 |      -0.1 |          2588.52 |             0 |
| ANTARA_OGD_ONLY |    0.013  |   -0.3216 |      -0.1 |          4230.23 |             0 |
| A-GEM           |    0.0636 |   -0.618  |      -0.1 |          2872.58 |             0 |
| EWC             |    0.0549 |   -0.5428 |      -0.1 |           816.47 |             0 |
| REPLAY          |    0.1808 |   -0.4437 |      -0.1 |           633.49 |             0 |
| NAIVE           |    0.0612 |   -0.6066 |      -0.1 |           729.91 |             0 |

## 🔬 Analysis vs. Hypotheses

### 1. Stability-Plasticity (BWT vs ACC)
*   **Target**: BWT > -0.05.
*   **Observation**: Check if ANTARA maintains higher stability than EWC without sacrificing accuracy.

### 2. Efficiency Claim (Step Time)
*   **Target**: ANTARA Step Time < A-GEM Step Time.
*   **Observation**: ANTARA's OGD should demonstrate lower latency than A-GEM's secondary projection pass.

### 3. Deliberative Synergy (Ablation)
*   **Target**: Full > RGW > OGD.
*   **Observation**: Intelligence in ANTARA is emergent from the System 1/System 2 interaction.
