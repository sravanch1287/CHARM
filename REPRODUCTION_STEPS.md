# Eight reproduction stages

For immediate published-result plots/tables, install `requirements.txt` and run
`python -m charm reproduce`. The following workflow generates **new** training
and evaluation results. Use Python 3.11 and `requirements-training.txt`.

## 01. Prepare data

Obtain the original Boston `pathTable.csv` as described in data/README.md.
Each experiment validates its hash. `--data-dir` names the containing folder;
`--workspace` controls the new run directory (default `runs/reproduction`).
The CLI defaults to CPU; use `--device cuda` for an available CUDA installation.

## 02. Train five seeds

```bash
python -m charm train --all --data-dir "PATH/TO/BOSTON"
```

Runs EPA, DPA2, SLAPP and seeds 1–5. For one model use `--method SLAPP --seed 1`.
Paper settings are 800 configured episodes and 80 decisions per episode. The
preserved loop logs indices 0–800 inclusive. `--episodes 6 --steps 80` is a short
integration test, not a paper training run. Existing run directories are refused.
Each run saves its policy, full-state checkpoints, configuration and logs.

## 03. Evaluate models

```bash
python -m charm evaluate --all --data-dir "PATH/TO/BOSTON"
```

Loads newly trained final policies. Default: five demand profiles, 50 episodes
per profile, 80 steps. Demands are installed before reset and held fixed within
an episode. Outputs are under `runs/reproduction/evaluation/five_demand_five_seed`.

## 04. Run Tier 2

```bash
python -m charm tier2 --data-dir "PATH/TO/BOSTON"
```

Exact fixed-SLAPP joint enumeration with parity checks. `--dry-count` counts
feasible assignments without computing channels. The static maximum is
80.104 Gbps with three CUs for the audited input. Historical MATLAB SA/GA/Greedy
comparisons are not replaced with a different algorithm in this release.

## 05. Run physical-layer comparisons

```bash
python -m charm phy --data-dir "PATH/TO/BOSTON"
```

Validates the SINR and iterative solvers, then evaluates the frozen new SLAPP
policies on paired profiles. These five-demand comparisons differ from the
U[4,8] post-training Table 5 protocol. Runtime ratios depend on hardware.

## 06. Generate tables

```bash
python -m charm posttraining --data-dir "PATH/TO/BOSTON"
python -m charm tables --from-runs runs/reproduction --output outputs/new_tables
```

`posttraining` evaluates all 15 new models over 50 common U[4,8] episodes and
scores paired SINR/iterative variants for SLAPP. `tables` aggregates Tables 5–7
from these results and the stage-03 evaluation. Omit `--from-runs` to use the
bundled published-run metrics instead. Output: CSV, LaTeX and numerical checks.

## 07. Generate figures

```bash
python -m charm figures --from-runs runs/reproduction --output outputs/new_figures
```

Requires all 15 complete training runs. Produces the Figure 5 three-panel and
Figure 6 four-panel plots plus exact plotted means/SDs in CSV. Omit `--from-runs`
to replot the published numerical traces without training or external input.

## 08. Verify expected results

```bash
python -m charm verify --from-runs runs/reproduction --output outputs/new_checks
python -m unittest discover -s tests -v
```

The verifier checks all 240 printed mean/SD values from Tables 5–7 against their
accepted rounding precision. New stochastic runs may differ; failure reports are
retained and tolerances are not silently loosened. Use `python -m charm verify`
to verify the bundled published data. New output directories must be empty.
Full experiment replay and unavailable Figure 4 inputs remain distinct from the
tested quick reproduction path; see docs/VALIDATION.md and docs/RESULT_MAP.md.
