# data_provider — datasets and loaders

`data_factory.py` maps `args.data` (this project uses `"custom"`) to
`Dataset_Custom` in `data_loader.py`, which reads one per-station CSV,
splits it chronologically by `args.ratios` (default 0.7/0.1/0.2), and yields
sliding windows of `seq_len` hours with the next `pred_len` hours as the
target (target column: `users`, mode `MS` — multivariate in, univariate
out). Loader policy: train is shuffled with `drop_last=True`; validation and
test are deterministic, unshuffled and complete (`drop_last=False`), so
early stopping and reported metrics always see every window.
`Dataset_Pred` serves `predict.py` with the final window of a CSV.
