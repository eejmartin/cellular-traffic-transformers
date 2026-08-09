#!/usr/bin/env bash
# Phase 2 base campaigns: 4 models x {Set 1 (91 bs), Set 2 (first 100 big_data)}
# x 5 seeds, no transfer. Every run lands in its own
# master_thesis_final/result_data_*_s<seed>_<ts>/ folder.
# Run detached:  nohup bash scripts/run_phase2_base.sh > master_thesis_final/phase2_base.log 2>&1 &
cd "$(dirname "$0")/.."

SEEDS="2021 2022 2023 2024 2025"

for seed in $SEEDS; do
    echo "=== [$(date '+%F %T')] Set 1 baseline, seed $seed ==="
    python run.py -s "$seed" || echo "FAILED: bs seed $seed"
    echo "=== [$(date '+%F %T')] Set 2 baseline, seed $seed ==="
    python run.py -f big_data -l 100 -s "$seed" || echo "FAILED: big_data seed $seed"
done
echo "=== [$(date '+%F %T')] phase 2 base campaigns done ==="
