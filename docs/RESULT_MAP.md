# Paper coverage

| Item | Public command/source | Scope |
|---|---|---|
| Figures 1–3 | Architectural and scenario illustrations | Editable artwork is not included. |
| Figure 4 | Original four-city MATLAB/ray-tracing workflow | The exact four-city generator and input lineage are not established in this release. No synthetic substitute is supplied. |
| Figure 5 | `python -m charm figures` | Three panels: convergence, sum rate, QoS. Uses all 15 reported training runs, centered 20-episode smoothing per seed, followed by mean and sample SD across seeds. |
| Figure 6 | `python -m charm figures` | Safety, validity, NRF, and timestep-level joint raw-action compliance, in accepted panel order. |
| Tables 1–4 | Experiment specifications | City metadata, physical parameters, hyperparameters and demand distributions. These are not simulation-output tables. Implemented training settings are recorded in `configs/training.json`. |
| Table 5 | `python -m charm tables` | Reaggregates saved per-training-seed post-training U[4,8] results. Five policies per method; SINR-PA and SLAPP-Iter reuse SLAPP policies. |
| Tables 6–7 | `python -m charm tables` | Reaggregates five-seed per-scenario means from 50 episodes/profile, 80 decisions/episode. Outputs all accepted mean and SD values. |
| Tier 2 exact 80.104-Gbps bound | `python -m charm tier2 --data-dir PATH` | New computation on the exact Boston instance; 281,400 capacity-feasible assignments. Requires external data. |
| SINR/iterative ablations | `python -m charm phy --data-dir PATH` | New fixed-policy comparisons after training all models. Requires external data/checkpoints. |

`reproduce` regenerates published-run numerical results; it does not retrain.
`train` and `evaluate` generate new results, which may differ across hardware or
dependencies. A full 15-model retraining was not performed during packaging.
The final article's typesetting/fonts are not reproduced byte-for-byte.

The compact training dataset was derived from the reported training pickles:
15 models, 801 logged episode indices (0–800), 80 decisions per episode. Joint
compliance was computed from the timestep AND of zero safety, validity and NRF
corrections, not a product of marginal rates. Hashes are in paper_data/provenance.json.

The production modules preserve the inspected campaign's numerical definitions.
The inspected trainer matches a later provenance record but differs from an
earlier frozen-manifest hash. This release does not claim byte-identical recovery
of every historical training job. Source fingerprints are in configs/source_provenance.json.

Boston single-seed reviewer-response experiments and obsolete code branches are
not included. The project license and any third-party data permissions should
be set by the rights holders; this packaging step does not grant them.
