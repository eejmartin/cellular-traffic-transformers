# Model Setup & Parameter Reference

This project compares four Transformer architectures on cellular-traffic
forecasting (predicting the number of `users` at a base station one hour
ahead). One model instance is trained per base-station CSV so results are
comparable per station and across architectures.

Reference implementations the models were adapted from:

| Model | Paper idea (one line) | Original repository |
|---|---|---|
| PatchTST (`TST`) | Split each channel into patches, encode patches with a vanilla Transformer encoder, channel-independent | https://github.com/yuqinie98/patchtst |
| CATS | Decoder-only cross-attention: learnable "dummy" future queries attend to past patches (no self-attention) | https://github.com/dongbeank/cats |
| DeformableTST | Hierarchical Transformer with deformable attention — each query attends to a small set of learned, data-dependent sampling locations | https://github.com/luodhhh/DeformableTST |
| Peri-midFormer (`PerimidFormer`) | Decomposes the series into a pyramid of periodic components (via FFT) and attends within/across pyramid levels | https://github.com/QiangWu-AI/Peri-midFormer |

## How to run

```bash
conda activate py312
python run.py
```

`run.py` trains, tests and predicts **every model in `models` × every CSV in
`data/{big_bs,medium_bs,small_bs}/`**, then plots true-vs-predicted curves.
All configuration lives in two dictionaries at the top of `run.py`:

- `SHARED_ARGS` — identical for all four models. Change a value here and it
  changes for every model, keeping the comparison valid.
- `MODEL_CONFIGS` — per-model settings, kept to what is architecture-specific.

The only command-line options are dataset selection (`-f bs|big_data`), the
station-count cap (`-l N`) and the transfer-learning switch (`-t`, see
"Experimental design" item 4); everything else lives in the dictionaries,
which override the argparse defaults in `utils/parser.py`.

## Experimental design (what makes the comparison valid)

1. **Same data pipeline** for all models: same CSVs, same 0.7/0.1/0.2
   chronological train/val/test split, same windowing (`seq_len=24 → pred_len=1`),
   same batch size and loader settings.
2. **Same training protocol**: MSE loss, Adam, learning rate 1e-4, up to 100
   epochs, early stopping on validation loss (patience 50), `type3` LR decay.
   - Exception (deliberate, marked "paper recipe" in `run.py`): DeformableTST
     trains with AdamW (weight decay 0.05) and 5-epoch linear warmup + cosine
     annealing, because the original authors publish it as part of the method.
     Delete those three keys from its `MODEL_CONFIGS` entry to force the fully
     identical shared protocol.
3. **Same evaluation**: loss/metrics are always computed on the last
   `pred_len` steps of the target channel only; every model writes the same
   artifacts (`metrics.npy`, `pred.npy`, `true.npy`, `real_prediction.npy`)
   under `result_data/results/<setting>/`, plus a line in `result_data/result.txt`.
4. **Independent runs**: each base station trains from scratch. Warm-starting
   from the previous station ("transfer") is off by default; enable it with
   the `-t/--transfer` CLI flag (works with both `-f bs` and `-f big_data`)
   or by setting `"transfer": True` in `SHARED_ARGS`. Either way it applies
   to **all** models (never to only some, or the comparison breaks). The
   transfer checkpoint (`transfer_model_<model>.pth`) is written inside the
   run's own timestamped folder, so fresh runs always start clean; only the
   legacy flat `result_data/` layout requires deleting stale
   `transfer_model_*.pth` files first.

   `--transfer-mode` (implies `-t`) picks the strategy, implemented in
   `utils/transfer_strategies.py`: `sequential` (file order, default),
   `ordered` (greedy nearest-neighbour chain over z-normalized 24h `users`
   profiles, started at the highest-volume station), `grouped` (KMeans over
   the same profiles, `--groups` clusters, independent chain per cluster —
   the checkpoint is deleted at cluster boundaries), `federated` (FedAvg:
   `--fed-rounds` rounds of `--fed-epochs` local epochs per station from
   shared global weights, sample-weighted averaging; the global weights are
   saved as the transfer checkpoint and every station fine-tunes from them —
   `transfer_save=False` keeps the per-station trainings from overwriting
   it), `combined` (all three: FedAvg pretraining, then one
   similarity-ordered chain per KMeans cluster, each chain seeded — and at
   every cluster boundary re-seeded — from the federated global weights;
   within a chain the checkpoint updates normally). Profiles are computed on
   the training split only.
5. **One training loop for everything**: `exp/exp_main.py::Exp_Main` handles
   all four models. The only per-model branch is how `forward` is invoked
   (Peri-midFormer additionally receives the time-of-day features and its
   output of length `label_len+pred_len` is sliced to the last `pred_len`
   steps, exactly like the other models).

## Data

Two per-station datasets are available, selected with `run.py`'s only CLI
flag (`-f`); everything else about training is identical in both modes and
every station always gets its own model instance:

**`-f bs` (default)** — the 91 committed CSVs in
`data/{big_bs,medium_bs,small_bs}/bs_<id>.csv`, ~192 hourly rows each:

```
time_hour, packets, bytes, users
```

With `seq_len=24` the 0.7/0.1/0.2 split gives ~110 training windows,
~20 validation windows, ~38 test windows per station.

**`-f big_data`** — the ~17,000 stations parsed from the raw NetData dumps
by `parse_big_data.py` into `data/big_data/parsed/concat/{4G,5G}/bs_<id>.csv`
(48 hourly rows: the representative weekday re-dated to Friday 2024-01-05
followed by the weekend day Saturday 2024-01-06, so the series is
continuous). Columns are `time_hour, prb, bytes, users` (the raw dumps have
no packet counts; PRB usage ratio is the third feature — the pipeline only
cares that there are 3 feature columns and a `users` target). Because the
files are short, this mode overrides `seq_len` to 12 (and `fmap_size` to 12
for DeformableTST): 21 training / 6 validation / 9 test windows per station.
Use `-l N` to train a subset — the full set is 17k stations × 4 models.

`features="MS"` means all 3 numeric columns are model input and only `users`
(always reordered to the last column) is predicted.

**Scaling.** `"scale": False` keeps raw units: metrics and plots are directly
interpretable as "number of users". This is safe here because all four models
normalize each input window internally (RevIN / instance normalization) and
de-normalize their output. Set `"scale": True` to standardize features with a
`StandardScaler` fit on the train split (the original repos' default); metrics
are then in standardized units. Whichever you choose, it applies to all
models equally.

## Shared parameters (`SHARED_ARGS`)

### Task & data

| Parameter | Value | Meaning |
|---|---|---|
| `data` | `custom` | selects `Dataset_Custom` in `data_provider/` |
| `features` | `MS` | multivariate input → univariate output |
| `target` | `users` | the predicted column |
| `freq` | `h` | hourly time features (hour, weekday, day, month) |
| `ratios` | `0.7,0.1,0.2` | chronological train/val/test fractions |
| `scale` | `False` | see "Scaling" above |
| `seq_len` | 24 | input window: one day of history |
| `label_len` | 1 | decoder warm-up steps (kept for pipeline compatibility; only Peri-midFormer's output length depends on it) |
| `pred_len` | 1 | forecast horizon: next hour |
| `enc_in` / `dec_in` / `n_vars` / `chan_in` | 3 | input-channel count; the four repos each read a different attribute name, so all four are set to 3 |

### Architecture scale (shared across models)

| Parameter | Value | Meaning |
|---|---|---|
| `d_model` | 32 | embedding width (validation-grid winner; small on purpose — ~110 training samples per station) |
| `n_heads` | 4 | attention heads |
| `e_layers` | 3 | PatchTST encoder depth |
| `d_layers` | 3 | CATS layer count (its only stack; matches `e_layers`) |
| `d_ff` | 256 | feed-forward width (8 × d_model) |
| `dropout` | 0.2 | main dropout (validation-grid winner) |
| `fc_dropout` / `head_dropout` | 0.05 / 0.0 | PatchTST head dropouts |
| `patch_len` / `stride` | 6 / 1 | patching for PatchTST & CATS: 20 overlapping 6-hour patches per 24-hour window (`stride` means something different for DeformableTST — see below) |
| `padding_patch` | `end` | replication-pad the sequence end before patching |
| `revin` | 1 | instance normalization on (PatchTST & DeformableTST) |
| `affine` / `subtract_last` | 0 / 0 | RevIN options: no learnable affine, subtract mean not last value |
| `decomposition` / `kernel_size` | 0 / 25 | PatchTST trend decomposition off; kernel kept for Peri-midFormer's decomposition |
| `individual` | 0 | one shared prediction head for all channels |

### Training protocol

| Parameter | Value | Meaning |
|---|---|---|
| `train_epochs` | 100 | maximum epochs |
| `batch_size` | 6 | → 18 gradient steps per epoch on ~110 windows |
| `patience` | 50 | early stopping: stop after 50 epochs without validation improvement; the checkpoint saved is always the **best** validation epoch |
| `learning_rate` | 5e-4 | base LR for every model (validation-grid winner) |
| `lradj` | `type3` | LR schedule: constant for 2 epochs, then ×0.9 each epoch |
| `optimizer` | `Adam` | `AdamW` (+`weight_decay`) only for DeformableTST |
| `warmup_epochs` | 0 | linear LR warmup epochs; only used with `lradj="cos"` |
| `weight_decay` | 0.05 | only applied when optimizer is AdamW |
| `pct_start` | 0.3 | OneCycleLR ramp fraction; only used with `lradj="TST"` |
| `num_workers` | 0 | dataloader workers; 0 is fastest for these tiny files |
| `embed` | `timeF` | continuous time-feature encoding (used by Peri-midFormer) |
| `transfer` | `False` | train each station from scratch; enable warm-starting with the `-t` CLI flag (see above) |
| `output_attention` | `False` | must stay `False`: none of the four models returns attention maps |

Available `lradj` schedules (`utils/tools.py::adjust_learning_rate` plus the
two scheduler-based options in `Exp_Main.train`):

- `type1` — halve every epoch; `type2` — fixed table; `type3` — constant 2
  epochs then ×0.9/epoch; `constant`; `3/4/5/6` — step drops.
- `TST` — OneCycleLR stepped every iteration (PatchTST's alternative recipe).
- `cos` — linear warmup for `warmup_epochs`, then cosine annealing to 0 over
  the remaining epochs (DeformableTST's recipe).

## Per-model configuration (`MODEL_CONFIGS`)

### PatchTST (`TST`)

Uses only shared parameters. The 24-step window is split into 20 overlapping
patches (`patch_len=6, stride=1, padding_patch=end`), each linearly embedded
to `d_model=32` and fed through a 3-layer channel-independent Transformer
encoder; a flatten head maps the encoded patches to the 1-step forecast.
RevIN normalizes each window and de-normalizes the output.

### CATS

| Parameter | Value | Meaning |
|---|---|---|
| `d_layers` | 3 (shared) | number of cross-attention layers — CATS has no encoder; this is its whole depth |
| `QAM_start` / `QAM_end` | 0.1 / 0.3 (parser defaults) | query-adaptive masking: dropout probability ramp applied across future queries during training (original repo defaults) |
| `query_independence` | `True` (parser default) | one shared learnable query set across channels |
| `patch_len` / `stride` | 6 / 1 (shared) | past patches, same as PatchTST |

CATS builds `ceil(pred_len/patch_len)=1` learnable future-patch query, which
cross-attends to the 20 past patches; the projection outputs one 6-step patch
and the first `pred_len=1` step is kept. Note: CATS reads the channel count
from `dec_in` and its depth from `d_layers` (that is why both are in
`SHARED_ARGS`).

### DeformableTST

The original 4-stage hierarchy halves the sequence between stages — a 24-step
input would shrink to 3 features. For short inputs the original authors
themselves switch to a **single stage** (their ILI script: `seq_len=24`,
`stem_ratio=1`, one stage), which is what we use, sized to match the shared
scale:

| Parameter | Value | Meaning |
|---|---|---|
| `stem_ratio` | 1 | no downsampling in the input stem (kernel-1 conv) |
| `down_ratio` | 2 | inter-stage downsampling factor (unused with one stage) |
| `dims` | `[32]` | channel width per stage = `d_model` |
| `depths` | `[3]` | Transformer blocks in the stage = `e_layers` |
| `stage_spec` | `[["D","D","D"]]` | block types: three Deformable-attention blocks |
| `heads` | `[4]` | attention heads = `n_heads` |
| `expansion` | 8 | FFN width factor: 16×8 = 128 = `d_ff` |
| `fmap_size` | 24 | sizes the relative-position-bias table (interpolated at runtime, so it only needs to be ≈ `seq_len`) |
| `ksize` / `stride` | `[3]` / `[1]` | kernel/stride of the offset sub-network that predicts sampling locations (`stride` here is per-stage and unrelated to patching) |
| `n_groups` | `[2]` | offset groups (must divide `heads`) |
| `drop_path_rate` | 0.3 | stochastic depth (original default) |
| `use_lpu` / `local_kernel_size` | `[1]` / `[3]` | local perception unit (depthwise conv) before each block |
| `use_pe`, `dwc_pe`, `fixed_pe`, `log_cpb` | `[1]`,`[0]`,`[0]`,`[0]` | learned relative position bias (default variant) |
| `layer_scale_value` | `[-1]` | −1 disables LayerScale |
| `offset_range_factor` / `no_off` | `[-1]` / `[0]` | unrestricted offsets; offsets enabled |
| `use_dwc_mlp`, `window_size`, `nat_ksize` | `[1]`,`[3]`,`[3]` | FFN conv variant; window/neighborhood sizes (unused by `D` blocks) |
| `head_type` / `use_head_norm` | `Flatten` / 1 | flatten head with final LayerNorm |
| `revin_affine` / `revin_subtract_last` | 0 / 0 | RevIN options |
| **paper recipe:** `optimizer=AdamW`, `weight_decay=0.05`, `lradj=cos`, `warmup_epochs=5` | | published training setup; remove to use the shared protocol |

If you ever increase `seq_len` and want the multi-stage variant back:
`seq_len` must be a multiple of `stem_ratio × down_ratio^(num_stages−1)` and
every per-stage list (`dims`, `depths`, `heads`, `ksize`, `stride`, …) needs
one entry per stage.

### Peri-midFormer (`PerimidFormer`)

| Parameter | Value | Meaning |
|---|---|---|
| `layers` | 3 | pyramid-encoder depth = `e_layers` |
| `top_k` | 2 | number of dominant FFT frequencies used to build pyramid levels (small because 24 samples only resolve a few periods) |
| `moving_avg` | 25 | moving-average kernel for the seasonal/trend decomposition (trend is forecast by a separate linear layer) |
| `task_name` | `long_term_forecast` (shared) | selects the forecasting head |
| `embed=timeF`, `freq=h` (shared) | | time-of-day features added to the input before the pyramid is built |

The model decomposes each window into trend + seasonal parts, splits the
seasonal part into periodic components at `top_k` FFT-detected periods,
encodes the component pyramid with attention, and outputs
`label_len+pred_len` steps (the pipeline evaluates the last `pred_len`).
Note: its config keys `layers`/`chan_in` intentionally mirror the original
repo's naming.

## Outputs

Every invocation of `run.py` creates its own timestamped folder with one
subfolder per model:

```
result_data_<dataset>[_lN][_transfer-<mode>[-params]]_<YYYYMMDD_HHMMSS>/
    command.txt                                the exact run.py invocation
    TST/ | CATS/ | DeformableTST/ | PerimidFormer/
        checkpoints/<setting>/checkpoint.pth   best-validation weights
        configs/<setting>.json                 full training configuration (used by predict.py)
        results/<setting>/pred.npy, true.npy   test-set predictions/targets, (n_windows, pred_len, 1)
        results/<setting>/metrics.npy          [mae, mse, rmse, mape, mspe, rse, corr]
        results/<setting>/real_prediction.npy  forecast for the first hour after the end of the file
        result.txt                             appended mse / mae / rse per station
        test_results/<setting>/*.pdf           sample input+forecast plots
        plots/true_vs_pred_plot_<setting>.png  full test-series true-vs-predicted comparison
```

`<setting>` encodes the hyperparameters and station id. Runs never overwrite
each other; both `result_data/` (legacy) and `result_data_*/` are
git-ignored. The legacy flat `result_data/` layout is still what you get if
an `Exp_Main` is constructed without `results_dir` (e.g. from old scripts).

## Using trained models later (predict.py)

Each trained (model, station) pair is fully described by its config JSON +
checkpoint, so it can be reloaded any time without retraining:

```bash
# forecast the next hour for the station the model was trained on
python predict.py --config result_data_<ts>/TST/configs/<setting>.json

# forecast from a different CSV (must have the same column layout:
# time column + 2 feature columns + users)
python predict.py --config ... --data data/small_bs/bs_2613.csv

# save the forecast, or point at a moved checkpoint explicitly
python predict.py --config ... --output forecast.csv
python predict.py --config ... --checkpoint path/to/checkpoint.pth
```

`predict.py` rebuilds the exact architecture from the JSON, loads the best
checkpoint (found next to the config even if the run folder was moved), feeds
the **last `seq_len` (24) rows** of the CSV through the model and prints the
`pred_len` (1) predicted `users` values for the hours after the file ends.
From Python:

```python
from predict import load_trained_model, predict_next
exp, args = load_trained_model('result_data_<ts>/TST/configs/<setting>.json')
forecast = predict_next(exp, args, 'data/small_bs/bs_2613.csv')
```

Caveats:
- A checkpoint is station-specific (trained on ~110 windows of one station);
  applying it to another station is technically possible (the models
  instance-normalize), but treat that as transfer, not a calibrated forecast.
- The input CSV needs ≥ `seq_len` rows and the training channel count (3).
- If the model was trained with `scale=True`, predictions come out in
  standardized units; keep `scale=False` (the default) if you want raw user
  counts at inference time.

## Metrics

Metric definitions (`utils/metrics.py`): MAE, MSE, RMSE, MAPE, MSPE, RSE
(root relative squared error: RMSE normalized by the test-set standard
deviation, 1.0 = predicting the mean), CORR (mean per-feature correlation,
scaled by 0.01 as in the reference repos).

With `scale=False` all error metrics are in raw user counts. Because MSE is
scale-dependent, only compare metrics **between models on the same station**,
or use RSE for a scale-free view across stations.

## Bugs fixed in this revision (July 2026)

Pipeline (these invalidated or crashed cross-model runs):

1. `run.py` mutated one shared config dict across models — e.g. after
   DeformableTST, the next model inherited `lradj="cos"` and a per-stage
   `stride` list. Each run now gets a fresh merged dict.
2. `exp_main.py` applied DeformableTST's warmup + cosine schedule to **every**
   model (parser default `warmup_epochs=5`) *on top of* the `type3` decay —
   two conflicting schedulers per epoch. Exactly one schedule is active now,
   selected by `lradj`.
3. Transfer-learning weights were loaded whenever
   `result_data/transfer_model_<model>.pth` existed — silently warm-starting
   from a previous station (or a previous experiment). Now opt-in via
   `transfer` and off by default.
4. PerimidFormer could not run at all: it was routed to `Exp_Main` without a
   builder (crash), while the alternative `Exp_Long_Term_Forecast` had an
   incompatible `adjust_learning_rate` call (their repo's 3-arg version), a
   5-value `metric()` unpack against this repo's 7-value function, wrong
   checkpoint paths, `shutil.rmtree` on a non-existent folder, no `predict()`,
   and saved results where the plotting step never looked. All four models
   now share `Exp_Main`; the task-specific `exp_*` files are unused legacy.
5. `output_attention=True` made the experiment classes take `model(...)[0]`
   — for models returning a bare tensor this silently dropped the batch
   dimension. Set to `False` and no longer used in the unified loop.
6. `Dataset_Custom` parsed `ratios` but hardcoded the 0.7/0.2 split; the
   `scale` flag was not plumbed through `data_factory`. Both fixed.
7. Channel-count mismatches: CATS read `dec_in` (default 7) and DeformableTST
   `n_vars` (default 321) while the data has 3 channels. All four channel
   attributes are now set together in `SHARED_ARGS`.
8. DeformableTST used the 4-stage architecture on 24-step inputs (features
   reduced to length 3); switched to the original authors' single-stage
   short-input configuration.
9. Smaller fixes: `test_params_flop()` called without the model argument;
   `metrics.npy` save restored (CORR flattened to a scalar); `predict()`
   slices the model output to `pred_len` (Peri-midFormer outputs
   `label_len+pred_len`); plotting skips folders without `pred.npy`/`true.npy`
   and closes figures; `num_workers=0` (10 workers per epoch dominated
   runtime on 110-sample datasets).
