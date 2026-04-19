#!/bin/bash

# NeurIPS Experiment Watcher
# This script monitors a python process and restarts it if it crashes.
# Combined with ANTARA's task-level checkpointing, it ensures 100% uptime.

SCRIPT_NAME=$1

if [ -z "$SCRIPT_NAME" ]; then
    echo "Usage: ./run_with_retry.sh Phase_IV_Ablation/vanguard_full.py"
    exit 1
fi

until python3 "$SCRIPT_NAME"; do
    echo "Experiment crashed with exit code $?. Respawning and resuming from checkpoint..." >&2
    sleep 5
done

echo "Experiment completed successfully."
