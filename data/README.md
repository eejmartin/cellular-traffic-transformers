# data — expected dataset layout

No data files are distributed in this repository. The code expects the
following layout:

```
data/
├── big_bs/     bs_<id>.csv   Set 1: large stations  (30 files in the thesis)
├── medium_bs/  bs_<id>.csv   Set 1: medium stations (31)
├── small_bs/   bs_<id>.csv   Set 1: small stations  (30)
└── big_data/   Performance_{4G,5G}_{Weekday,Weekend}.txt   (Set 2, raw)
```

**Set 1** — one CSV per base station with columns
`time_hour, packets, bytes, users` (~192 hourly rows, 8 days). The three
folders encode the traffic-volume category; the loader reads them
round-robin. The thesis used a 91-station set provided by FEIT/UKIM for
research purposes; it is not redistributed here — any CSVs with the same
column layout work.

**Set 2** — download the public NetData dumps from
https://github.com/tsinghua-fib-lab/NetData into `big_data/` and run
`python parse_big_data.py`. It writes per-station hourly CSVs
(`time_hour, prb, bytes, users`) and the trainable 48-hour weekday+weekend
concatenation under `big_data/parsed/concat/{4G,5G}/` (17,327 stations; the
thesis experiments use the first 100). The whole folder is git-ignored.
