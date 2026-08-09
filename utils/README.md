# utils — shared utilities

| File | Contents |
|---|---|
| `parser.py` | argparse defaults; every key of the merged config dict from `run.py` is set on top of it, so values in `run.py` always win |
| `tools.py` | training utilities (early stopping, LR adjustment, warmup) + dataset discovery: `read_file_names_by_order()` (Set 1, round-robin over big/medium/small) and `read_big_data_file_names()` (Set 2 concat CSVs) |
| `metrics.py` | metric definitions: MAE/MSE/RMSE/MAPE/RSE/Pearson CORR plus sMAPE, WAPE, MASE, bias (ME), peak-MAE and underprediction rates; `metric_all()` returns the full named set saved as `metrics_ext.json` |
| `transfer_strategies.py` | station ordering (greedy similarity chain), KMeans grouping and FedAvg pretraining used by `--transfer-mode` |
| `timefeatures.py` | calendar feature encoding (hour-of-day, day-of-week, …) |
| `masking.py`, `losses.py` | attention masks and loss variants kept for the model implementations |
