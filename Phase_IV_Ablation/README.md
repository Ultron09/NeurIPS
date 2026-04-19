# Phase IV: The Ablation Autopsy

Reviewers often perceive advanced frameworks as "black boxes." This phase systematically dismantles ANTARA to prove the necessity of each component—System 1 (Reflexive) and System 2 (Deliberative)—and the OGD protection layer.

## Experimental Configurations

We conduct three distinct runs to identify the contribution of each module:

### Run 1: Full ANTARA (RGW + OGD)
- **Active Components**: Recursive Global Workspace (RGW) and Orthogonal Gradient Descent (OGD).
- **Hypothesis**: Provides maximum plasticity while maintaining rigid retention, achieving the SOTA balance between Forward (FWT) and Backward (BWT) transfer.

### Run 2: OGD Disabled
- **Active Components**: RGW only.
- **Hypothesis**: Projects significant Forward Transfer efficiency but suffers from catastrophic forgetting (BWT), proving the necessity of the projection layer for long-term survival.

### Run 3: RGW Disabled
- **Active Components**: OGD only (reflexive pass).
- **Hypothesis**: Maintains knowledge (low BWT) but lacks the reasoning capacity to adapt quickly to new, ambiguous tasks (low ACC/FWT), proving the necessity of the recursive workspace.

## Outcome
The architectural autopsy provides empirical proof that the "magic" of ANTARA lies in the intersection of its reflexive and deliberative mechanisms.

## Usage
Run the ablation orchestrator:
```bash
python ablation_runner.py
```
