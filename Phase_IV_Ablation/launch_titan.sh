#!/bin/bash
cd /teamspace/studios/this_studio/NeurIPS
export PYTHONUNBUFFERED=1
/home/zeus/miniconda3/envs/cloudspace/bin/python Phase_IV_Ablation/benchmark_runner.py --stages 7 --seeds 42 10 20 30 > /home/zeus/titan_breakthrough.log 2>&1
