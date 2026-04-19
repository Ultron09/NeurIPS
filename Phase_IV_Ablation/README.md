# Phase IV: The Architectural Autopsy (Ablation Gauntlet)

This is the flagship execution phase where the ANTARA framework is subjected to a 4-way comparison to prove the necessity of its cognitive components.

## 🏁 The Gauntlet Runs

We execute 4 distinct parallel scripts to populate the NeurIPS empirical tables:

1.  **`vanguard_full.py`**: The full ANTARA framework (Full protection).
2.  **`ablated_memory.py`**: Disables consolidation (Stability test).
3.  **`ablated_consciousness.py`**: Disables the Deliberative loop (Plasticity test).
4.  **`naive_control.py`**: Standard fine-tuning (Failure baseline).

## 🛡️ Self-Healing Architecture
Remote compute can be volatile. Phase IV implements **Atomic Resumption**. 
- Status is serialized to `results/checkpoints/` after every task.
- If a script crashes, simply restarting it will trigger the self-healing logic, resuming from the last completed task.

## 🚀 Parallel Launch Instructions
It is highly recommended to run these scripts on separate GPUs for maximum efficiency:

```bash
# Set PYTHONPATH to root before running
export PYTHONPATH=$PYTHONPATH:.

# Launch Vanguard
python Phase_IV_Ablation/vanguard_full.py &

# Launch Naive Control
python Phase_IV_Ablation/naive_control.py &
```

## 📊 Outputs
Results are stored in `./results/`:
- `gauntlet_[config].png`: Accuracy heatmap.
- `gauntlet_[config].json`: Serialized Ri,j matrix.
