#!/usr/bin/env bash
# Activate conda before set -u: conda deactivate hooks reference unset vars.
cd /home1/datahome/ilarroch/data/INES/CODE/waves2surfF/waves2surf_net
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seastatesenv

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
python train.py \
  --config configs/agulhas.json \
  --output-dir runs/agulhas_baseline
