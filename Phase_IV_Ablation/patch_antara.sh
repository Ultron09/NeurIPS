#!/bin/bash
CORE_PATH="/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/airborne_antara/core.py"
sed -i '/Anchoring Knowledge/d' $CORE_PATH
sed -i 's/.*copy.deepcopy(self.model).*/            self.teacher_model = self.model # BERSERKER/g' $CORE_PATH
echo "Antara Library Patched Successfully (V2)."
