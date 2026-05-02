# ANTARA: Recursive Introspective Architecture
# Official Reproducibility Environment for NeurIPS 2026

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the framework and ablation suite
COPY . .

# Default command to run the ablation suite
CMD ["python", "Phase_IV_Ablation/ablation_runner.py"]
