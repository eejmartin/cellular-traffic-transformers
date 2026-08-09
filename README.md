# Cellular Traffic Transformers

Code for the master's thesis **„Примена на трансформер архитектура во
оптимизација на радио пристапна мрежа на мобилни системи"** (*Application of
Transformer architectures in the optimization of the radio access network of
mobile systems*) — Martin Cvetanovski, Faculty of Electrical Engineering and
Information Technologies (FEIT), Ss. Cyril and Methodius University in Skopje.

The thesis forecasts the number of active users per base station (hourly
resolution, one hour ahead) and compares four modern Transformer
architectures under a fully identical, seeded, reproducible protocol:

| Model | Idea |
|---|---|
| **PatchTST** | patch tokenization + channel independence |
| **CATS** | cross-attention only, no self-attention |
| **DeformableTST** | deformable sparse attention, no patching |
| **Peri-midFormer** | FFT-detected periods, periodic pyramid |

On top of the per-station comparison the pipeline implements five
transfer-learning strategies between stations (`sequential`, `ordered`,
`grouped`, `federated`, `combined`), naive/linear baselines, ablations and a
validation-only hyperparameter search. The forecasts feed a conceptual O-RAN
integration (rApps/xApps for predictive resource allocation, load balancing
and cell sleeping) elaborated in the thesis.

## Datasets

```
data/
├── big_bs/     30 stations   bs_<id>.csv   ~192 hourly rows
├── medium_bs/  31 stations   time_hour, packets, bytes, users
├── small_bs/   30 stations   (Set 1 — committed in this repo)
└── big_data/   (Set 2 — NOT committed; public NetData dumps)
```

- **Set 1** (91 stations, grouped by traffic volume) was provided by
  FEIT/UKIM for research purposes and is redistributed here with permission.
- **Set 2**: download `Performance_{4G,5G}_{Weekday,Weekend}.txt` from the
  [Tsinghua FIB Lab NetData repository](https://github.com/tsinghua-fib-lab/NetData)
  into `data/big_data/`, then run `python parse_big_data.py` — it produces
  per-station 48-hour CSVs (`weekday + weekend`, 17,327 stations); the
  experiments use the first 100.

## Setup

```bash
conda create -n py312 python=3.12 pytorch torchvision torchaudio numpy \
    matplotlib pandas scikit-learn pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda activate py312
pip install -r requirements.txt
```

## Running

`run.py` is the single entry point (train → test → predict → plots for every
model × station). Hyperparameters live at the top of `run.py`
(`SHARED_ARGS` / `MODEL_CONFIGS`, documented in `models/README.md`); the CLI
selects the dataset, station cap, transfer strategy and seed:

```bash
python run.py                                      # Set 1, no transfer
python run.py -t                                   # + sequential transfer
python run.py --transfer-mode combined             # FedAvg + clustering + ordered chains
python run.py -f big_data -l 100 -s 2022           # Set 2, first 100 stations, seed 2022
```

Every run is deterministic: the seed (`-s`, default 2021) controls weight
init, dropout and shuffling, and identical commands reproduce bit-identical
metrics. Each run writes its own folder
`master_thesis_final/result_data_<command>_s<seed>_<timestamp>/<model>/` with
checkpoints, config JSONs, predictions and metrics; the exact command is
saved as `command.txt`.

Thesis experiment campaigns are scripted in `scripts/`:
`run_phase2_base.sh` (5-seed baselines), `run_phase2_transfer.sh` (transfer
strategies), `run_baselines.py` (naive/seasonal/linear references),
`run_ablations.py` (history length, seq_len, channels, horizons),
`hp_search.py` (validation-only grid search), `retest.py` (re-evaluate a
finished campaign from saved checkpoints).

## Analysis and reuse

- `analysis/run_analysis.py <run_dir> ...` — aggregate tables, charts and a
  report per campaign; `analysis/compare_runs.py` — paired comparison of two
  campaigns (e.g. with/without transfer). Model names are discovered
  dynamically, so the tooling survives new models and datasets.
- `predict.py --config <run_dir>/<model>/configs/<setting>.json` — rebuild a
  trained model from its saved config and forecast past the end of any CSV.
- `visualization/` — figure generation and thesis document build scripts.

## License

Code under the [MIT License](LICENSE). Set 1 data courtesy of FEIT/UKIM —
please credit the source if you use it.
