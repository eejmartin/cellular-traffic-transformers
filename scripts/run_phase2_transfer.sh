#!/usr/bin/env bash
# Phase 2 transfer campaigns:
#   sequential x 3 seeds x both datasets, then the four remaining strategies
#   (ordered / grouped / federated / combined) x 1 seed x both datasets.
# Run detached:  nohup bash scripts/run_phase2_transfer.sh > master_thesis_final/phase2_transfer.log 2>&1 &
cd "$(dirname "$0")/.."

for seed in 2021 2022 2023; do
    echo "=== [$(date '+%F %T')] Set 1 sequential transfer, seed $seed ==="
    python run.py -t -s "$seed" || echo "FAILED: bs seq seed $seed"
    echo "=== [$(date '+%F %T')] Set 2 sequential transfer, seed $seed ==="
    python run.py -f big_data -l 100 -t -s "$seed" || echo "FAILED: big_data seq seed $seed"
done

for mode_args in "ordered" "grouped --groups 3" "federated --fed-rounds 5 --fed-epochs 5" "combined --groups 3 --fed-rounds 5 --fed-epochs 5"; do
    echo "=== [$(date '+%F %T')] Set 1 transfer-mode $mode_args ==="
    python run.py --transfer-mode $mode_args -s 2021 || echo "FAILED: bs $mode_args"
    echo "=== [$(date '+%F %T')] Set 2 transfer-mode $mode_args ==="
    python run.py -f big_data -l 100 --transfer-mode $mode_args -s 2021 || echo "FAILED: big_data $mode_args"
done
echo "=== [$(date '+%F %T')] phase 2 transfer campaigns done ==="
