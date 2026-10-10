# CHARM

Supporting code for **“CHARM: A Constrained Hybrid Adaptive Reinforcement Learning Framework for Sub-THz Wireless Midhaul Networks.”**

CHARM combines reinforcement learning for CU deployment and DU association with beamforming and power allocation for sub-THz wireless midhaul networks.

## Repository contents

- `source/` — training, evaluation, plotting, and reference MATLAB code.
- `configs/` — experiment configurations.
- `environments/` — Python dependencies.
- `tools/` — experiment launcher and verification tools.
- `data/README.md` — input-data requirements.

## Getting started

Use Python 3.11 and install the dependencies:

```bash
pip install -r environments/requirements.txt
```

Follow [REPRODUCTION_STEPS.md](REPRODUCTION_STEPS.md) for data preparation, training, evaluation, and figure/table generation.

## Data availability

Datasets, pretrained models, and training logs are not included. The Python experiments require the original Boston `pathTable.csv`. Required external inputs are described in [data/README.md](data/README.md).
