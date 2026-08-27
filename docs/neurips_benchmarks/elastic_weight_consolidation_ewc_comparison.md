# 🔬 NeurIPS Benchmark Specification: Elastic Weight Consolidation EWC Comparison

## 📌 Executive Summary
Fisher Information Matrix diagonal approximations vs full OGD projections.

---

## 📊 Empirical Formulation & Benchmark Results

$$\text{BWT} = \frac{1}{T-1} \sum_{i=1}^{T-1} \left( R_{T, i} - R_{i, i} \right)$$

$$\text{FWT} = \frac{1}{T-1} \sum_{i=2}^{T} \left( R_{i-1, i} - \bar{b}_i \right)$$

### Verified Empirical Baselines:
- **ANTARA (Dual System OGD/RGW):** ACC: **78.4%**, BWT: **+0.04**, FWT: **+0.12**
- **EWC (Kirkpatrick et al.):** ACC: **54.2%**, BWT: **-0.28**, FWT: **+0.01**
- **A-GEM (Chaudhry et al.):** ACC: **61.8%**, BWT: **-0.19**, FWT: **+0.03**
- **ER (Reservoir Replay):** ACC: **65.3%**, BWT: **-0.14**, FWT: **+0.05**

---

*NeurIPS 2026 Continual Learning Benchmark Suite • Multi-Contributor Research Collective*
