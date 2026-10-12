# CHARM

Supporting code for **“CHARM: A Constrained Hybrid Adaptive Reinforcement Learning Framework for Sub-THz Wireless Midhaul Networks.”**

CHARM uses a constrained reinforcement-learning policy for CU deployment and DU
association, followed by beamforming and power allocation for wireless midhaul.

## Reproduce the reported plots and tables

From this folder, using Python 3.11:

```bash
python -m pip install -r requirements.txt
python -m charm reproduce
```

This writes **Figures 5–6** as PNG/PDF, **Tables 5–7** as CSV/LaTeX, and a
numerical verification report to `outputs/paper/`. It uses the bundled compact
numerical data from the reported five-seed experiments. No checkpoint, external
ray-tracing data, or new training is needed for this command. The numerical
values are reproduced; fonts and typesetting may differ from the final article.

## Where is the RL code?

| File | Purpose |
|---|---|
| `charm/thz_environment.py` | THz environment, rewards, constraints, and action projection |
| `charm/ppg_policy.py` | Actor/critic networks, PPO policy updates, and PPG auxiliary updates |
| `charm/train.py` | Training loop and checkpoint saving |
| `charm/physical_layer.py` | Channel construction, beamforming, power allocation, and rates |
| `charm/evaluate.py` | Five-demand-profile policy evaluation |
| `charm/power_allocation.py` | SINR-based and iterative PA comparisons |
| `charm/tier2.py` | Exact reduced-instance joint enumeration |
| `charm/publication.py` | Paper figures, tables, and numerical checks |

## Train a new model

Training needs the original Boston `pathTable.csv`, supplied separately as
described in [data/README.md](data/README.md).

```bash
python -m pip install -r requirements-training.txt
python -m charm train --method SLAPP --seed 1 --data-dir "PATH/TO/BOSTON"
python -m charm evaluate --method SLAPP --seed 1 --data-dir "PATH/TO/BOSTON"
```

Use `--all` to train/evaluate all three methods and five seeds. Training produces
new checkpoints locally; none are bundled. See [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md)
for the eight-stage workflow and commands for new-run figures and tables.

## Data and coverage

`paper_data/` contains small numeric plot/table inputs, not models or raw
ray-tracing data. The quick reproduction command covers Figures 5–6 and
Tables 5–7. Figures 1–3 are illustrations; Figure 4 requires the original
four-city workflow. [docs/RESULT_MAP.md](docs/RESULT_MAP.md) describes coverage.

## Tests

```bash
python -m unittest discover -s tests -v
```

> [!NOTE]
> **Acknowledgments**
>
> The material is based upon work supported by NSF grants 1955561, 2212565, 2323189, 2434113, and 2514415.


