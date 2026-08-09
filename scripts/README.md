# scripts — experiment campaigns

Thesis campaign runners; every campaign writes standard run folders under
`master_thesis_final/` (git-ignored), reusable by `analysis/`.

| Script | Purpose |
|---|---|
| `run_phase2_base.sh` | base campaigns: 4 models × both datasets × 5 seeds, no transfer |
| `run_phase2_transfer.sh` | sequential transfer × 3 seeds + ordered / grouped / federated / combined strategies |
| `run_baselines.py` | naive persistence, seasonal-naive and linear baselines evaluated on the exact model test windows (CPU-only) |
| `run_ablations.py` | ablations: seq_len 12 vs 24, users-only vs 3 channels, horizons 3/6/12, history length 48–192 h on 30 stratified stations |
| `hp_search.py` | validation-only hyperparameter grid search (12 configs × 9 stations × 4 models + DeformableTST protocol-parity arm) |
| `retest.py` | re-evaluate a finished campaign from its saved checkpoints without retraining |
