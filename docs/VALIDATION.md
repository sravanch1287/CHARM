# Executed validation

Validated on October 9, 2026, with Python 3.11 on Windows. Package versions are
recorded in TESTED_ENVIRONMENT.json. The quick reproduction workflow was also
run from a separate temporary folder whose path contains spaces, with no original
workspace in PYTHONPATH. Dependencies were already installed; a fresh installation
and the provided Linux CI workflow have not yet been executed.

Passed checks:

- All four unit tests, including 15 x 801 episode coverage, joint-compliance
  invariants, independent first-window smoothing, and 240 accepted table values.
- Actual generation of both multi-panel figures as PNG/PDF and three CSV/LaTeX
  tables, including exact plotted means and sample SDs as CSV.
- Visual inspection of both regenerated multi-panel figures.
- Core physical-layer, environment and policy function/class definitions match
  the inspected experiment source ASTs. Imports and paths were reorganized.
- Tier 2 dry enumeration counts 281,400 capacity-feasible assignments.
- Direct physical-layer computation of the known three-CU winning assignment:
  8 x 10.013 Gbps = 80.104 Gbps, checked at absolute tolerance 1e-11.
- A real CPU training run with 6 configured episodes x 80 steps (the preserved
  inclusive loop performs 560 decisions), crossing the 480-decision PPO update.
  Nonempty optimizer state was verified in the saved full-state checkpoint.
- Reload and evaluation of that newly saved policy across all five demand
  profiles, one episode/profile and five decisions/episode.
- No hardcoded author-machine paths or assistant-name references in public files.

Not claimed: a new 15-model, 800-episode full campaign; full policy-evaluation
equivalence across hardware; the complete Tier 2 search or PA comparison campaign;
exact publisher typography; original four-city Figure 4 regeneration; or a
verified historical dependency lock. Short training is an integration test, not
an estimate of the published final performance.
