# data — per-station datasets

**Set 1 (committed):** `big_bs/` (30 stations), `medium_bs/` (31),
`small_bs/` (30) — one `bs_<id>.csv` per base station, ~192 hourly rows with
columns `time_hour, packets, bytes, users`. Categories reflect traffic
volume. Provided by FEIT/UKIM for research purposes.

**Set 2 (not committed):** place the public NetData dumps
`Performance_{4G,5G}_{Weekday,Weekend}.txt`
(https://github.com/tsinghua-fib-lab/NetData) into `big_data/` and run
`python parse_big_data.py` — it writes per-station hourly CSVs
(`time_hour, prb, bytes, users`) and the trainable 48-hour
weekday+weekend concatenation under `big_data/parsed/concat/{4G,5G}/`.
The whole `big_data/` folder is git-ignored.
