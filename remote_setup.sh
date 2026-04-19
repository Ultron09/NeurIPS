#!/bin/bash

# remote_setup.sh
# Automation script for setting up the ANTARA NeurIPS Evaluation Suite on a remote machine.

echo "--- Starting ANTARA Suite Setup ---"

# 1. Update and install basic dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev

# 2. Install Python requirements
echo "--- Installing Python dependencies ---"
pip install -r requirements.txt

# 3. Verify installation
echo "--- Verifying airborne-antara installation ---"
python3 -c "import airborne_antara; print('ANTARA Version:', airborne_antara.__version__)"

echo "--- Setup Complete ---"
echo "You can now execute experiments in the Phase directories."
