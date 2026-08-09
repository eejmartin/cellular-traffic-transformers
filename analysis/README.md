# Results analysis toolkit

Reusable post-training analysis for the `result_data_<timestamp>/` folders
produced by `run.py`: parses metrics, builds comparison charts, plots
true-vs-predicted series and writes a markdown report. Works on any future
run — with different datasets, more stations or new/renamed models.

## Quick start

```bash
# single run
python analysis/run_analysis.py result_data_20260705_131934

# compare two runs side by side (e.g. bs vs big_data), Macedonian labels
python analysis/run_analysis.py \
    master_thesis/result_data_20260705_131934 \
    master_thesis/result_data_20260706_072702 \
    --names "Set 1 (bs)" "Set 2 (big_data)" --lang mk -o analysis_results
```

Run from the repository root. Outputs go to `./analysis_results/` by default
(override with `-o`).

## What it produces

| Output | Content |
|---|---|
| `metrics_all.csv` | tidy table: one row per run × model × station (MSE, MAE, RSE + RMSE/MAPE/CORR when `metrics.npy` exists) |
| `summary_by_model.csv` | mean/median metrics, share of stations with RSE≥1, win counts |
| `bar_rse.png`, `bar_mae.png` | mean + median per model, one panel per run |
| `box_rse.png` | per-station RSE distribution per model (RSE=1 naive-mean line) |
| `wins.png` | number of stations where each model has the lowest MSE |
| `group_breakdown.png` | mean RSE per station group — big/medium/small category or 4G/5G technology (skipped if only one group) |
| `pred_grid_<run>_<station>.png` | true vs predicted per model, for representative stations |
| `pred_overlay_<run>_<station>.png` | all models overlaid on the true series |
| `params.png` | trainable parameters per model (needs torch; skip with `--no-params`) |
| `analysis_report.md` | tables + key findings + all figures in one document |

Representative stations are auto-selected as best / median / worst by median
RSE across models; override with `--stations bs_6260 bs_5113 ...`.

## How model names are resolved

Nothing is hardcoded to the current four models:

1. the models in a run are whatever subfolders of the run directory contain a
   `result.txt`;
2. display labels come from the file names in `models/` via the
   `MODEL_CLASSES` mapping in `exp/exp_main.py` (e.g. run key `TST` →
   `models/PatchTST.py` → label *PatchTST*); a key without a mapping keeps its
   folder name;
3. colors are assigned to labels in stable sorted order, so a model keeps its
   color across figures and runs.

Adding a new model (`models/MyModel.py` + `MODEL_CLASSES` entry + `run.py`
models list) requires **no change** in this folder.

## How stations are grouped

The `group` column drives `group_breakdown.png`: a technology prefix embedded
in the station name (`4G_bs_7` → `4G`) or, failing that, the `data/` subfolder
containing the station CSV (`big_bs`/`medium_bs`/`small_bs`; configurable with
`--data-root`). Runs with a single group simply skip the breakdown chart.

## Files

- `run_analysis.py` — CLI entry point orchestrating everything
- `metrics_parser.py` — result.txt/metrics.npy/configs → tidy DataFrame + summary
- `charts.py` — aggregate comparison charts (bars, boxplots, wins, breakdown, params)
- `pred_plots.py` — per-station true-vs-predicted grids and overlays
- `params.py` — optional parameter counting by rebuilding models from config JSONs
- `report.py` — markdown report writer
- `models_registry.py` — model/label/color discovery (see above)

`--lang en|mk` switches all chart labels and the report between English and
Macedonian (the thesis figures use `mk`).
