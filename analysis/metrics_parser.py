"""Parse run results (result.txt + metrics.npy + configs) into a DataFrame.

One row per (run, model, station) with columns:
    run, model, label, station, group, setting, mse, mae, rse
and, when results/<setting>/metrics.npy exists, additionally:
    rmse, mape, mspe, corr

``group`` is a best-effort partition of the stations, used for breakdown
charts:
  * a technology/prefix embedded in the station name (e.g. 4G_bs_7 -> '4G'), or
  * the data subfolder the station CSV lives in (big_bs/medium_bs/small_bs), or
  * 'all' when nothing better is known.
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd

from models_registry import discover_run_models, label_for

_METRIC_LINE = re.compile(
    r"mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+),\s*rse:([\d.eE+-]+)")
# metrics.npy layout written by exp_main.test()
_NPY_FIELDS = ["mae", "mse", "rmse", "mape", "mspe", "rse", "corr"]


def _station_from_setting(setting, data_path, root_path):
    """Recover the station id from a setting string.

    run.py builds the station file_name as the CSV stem, prefixed with the
    data subfolder name when the same stem can occur in several subfolders
    (big_data: '4G_' / '5G_'). Reproduce exactly that rule so station names
    align across models.
    """
    stem = os.path.splitext(os.path.basename(data_path))[0] if data_path else None
    if stem:
        folder = os.path.basename((root_path or "").rstrip("/\\"))
        if folder and setting.endswith(f"_{folder}_{stem}"):
            return f"{folder}_{stem}"
        return stem
    # no config available: fall back to a generic '..._<name>_<number>' suffix
    m = re.search(r"([A-Za-z0-9]+_[A-Za-z]+_\d+)$", setting)
    return m.group(1) if m else setting


def _load_configs(model_dir):
    """setting -> config dict, from the configs/*.json saved by run.py."""
    configs = {}
    for path in glob.glob(os.path.join(model_dir, "configs", "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            configs[cfg.get("setting", os.path.splitext(os.path.basename(path))[0])] = cfg
        except (OSError, json.JSONDecodeError):
            continue
    return configs


def _station_group(station, root_path, data_root):
    """Partition key for breakdown charts (see module docstring)."""
    m = re.match(r"([A-Za-z0-9]+)_(.+_\d+)$", station)
    base_csv = station + ".csv"
    if m and root_path and os.path.basename(root_path or "") == m.group(1):
        return m.group(1)          # technology prefix, e.g. 4G / 5G
    if data_root and os.path.isdir(data_root):
        for sub in sorted(os.listdir(data_root)):
            if os.path.isfile(os.path.join(data_root, sub, base_csv)):
                return sub          # e.g. big_bs / medium_bs / small_bs
    if m:
        return m.group(1)
    return "all"


def parse_run(run_dir, run_name=None, data_root=None):
    """Parse one result_data_* directory into a tidy DataFrame."""
    run_dir = os.path.abspath(run_dir)
    run_name = run_name or os.path.basename(run_dir.rstrip("/"))
    rows = []
    for model in discover_run_models(run_dir):
        model_dir = os.path.join(run_dir, model)
        configs = _load_configs(model_dir)
        label = label_for(model)

        with open(os.path.join(model_dir, "result.txt"), encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for i in range(0, len(lines) - 1, 2):
            setting, metric_line = lines[i], lines[i + 1]
            m = _METRIC_LINE.match(metric_line)
            if not m:
                continue
            mse, mae, rse = map(float, m.groups())
            cfg = configs.get(setting, {})
            station = _station_from_setting(setting, cfg.get("data_path"), cfg.get("root_path"))
            group = _station_group(station, cfg.get("root_path"), data_root)
            row = dict(run=run_name, model=model, label=label, station=station,
                       group=group, setting=setting, mse=mse, mae=mae, rse=rse)

            npy = os.path.join(model_dir, "results", setting, "metrics.npy")
            if os.path.exists(npy):
                try:
                    vals = np.load(npy)
                    for name, val in zip(_NPY_FIELDS, vals):
                        if name not in ("mae", "mse", "rse"):  # keep result.txt values
                            row[name] = float(val)
                except (OSError, ValueError):
                    pass
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No results parsed from {run_dir!r}")
    dup = df.duplicated(subset=["run", "model", "station"], keep="last")
    if dup.any():
        # result.txt is append-only: a re-run of the same station leaves an old
        # line behind — keep the most recent entry.
        df = df[~dup]
    return df.reset_index(drop=True)


def parse_runs(run_dirs, run_names=None, data_root=None):
    """Parse several run directories into one DataFrame."""
    run_names = run_names or [None] * len(run_dirs)
    frames = [parse_run(d, n, data_root) for d, n in zip(run_dirs, run_names)]
    return pd.concat(frames, ignore_index=True)


def summarize(df):
    """Aggregate table per (run, model): mean/median of key metrics, share of
    stations with RSE>=1 and win counts (lowest MSE per station)."""
    parts = []
    for run, sub in df.groupby("run", sort=False):
        agg = sub.groupby("label").agg(
            stations=("station", "nunique"),
            mse_mean=("mse", "mean"), mse_median=("mse", "median"),
            mae_mean=("mae", "mean"), mae_median=("mae", "median"),
            rse_mean=("rse", "mean"), rse_median=("rse", "median"),
            rse_ge_1=("rse", lambda s: float((s >= 1).mean())),
        )
        wins = (sub.pivot_table(index="station", columns="label", values="mse")
                   .idxmin(axis=1).value_counts())
        agg["wins"] = wins.reindex(agg.index).fillna(0).astype(int)
        agg.insert(0, "run", run)
        parts.append(agg.reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["run", "rse_mean"]).reset_index(drop=True)
