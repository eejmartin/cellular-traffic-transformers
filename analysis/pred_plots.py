"""True-vs-predicted figures per station, from saved pred.npy / true.npy.

For each selected station two figures are produced:
  * a grid with one panel per model (true vs predicted + per-panel MSE/MAE)
  * an overlay of all models' predictions against the true series

Stations can be given explicitly; otherwise representative ones are picked
automatically: the best, the median and the worst station according to the
median RSE across models — showing the whole quality range, not just the
prettiest example.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models_registry import assign_colors

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"

STRINGS = {
    "en": {"true": "true values", "pred": "prediction",
           "xlabel": "test window (hour)", "ylabel": "users",
           "grid_title": "{run}, station {station}: true vs predicted (test set)",
           "overlay_title": "{run}, station {station}: all models vs true values"},
    "mk": {"true": "реални вредности", "pred": "прогноза",
           "xlabel": "тест прозорец (час)", "ylabel": "корисници",
           "grid_title": "{run}, станица {station}: реални наспроти предвидени вредности",
           "overlay_title": "{run}, станица {station}: сите модели наспроти реалните вредности"},
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def _style(ax):
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def select_stations(df_run, n=3):
    """Pick representative stations: best / median / worst by the median RSE
    across models (falls back to fewer when the run has few stations)."""
    med = df_run.groupby("station")["rse"].median().sort_values()
    if len(med) <= n:
        return list(med.index)
    idx = np.linspace(0, len(med) - 1, n).round().astype(int)
    return [med.index[i] for i in idx]


def _load_pred_true(run_dir, df_run, label, station):
    """Locate results/<setting>/{pred,true}.npy via the setting recorded in
    the DataFrame."""
    sub = df_run[(df_run.label == label) & (df_run.station == station)]
    if sub.empty:
        return None
    setting, model_key = sub.iloc[0]["setting"], sub.iloc[0]["model"]
    folder = os.path.join(run_dir, model_key, "results", setting)
    p, t = os.path.join(folder, "pred.npy"), os.path.join(folder, "true.npy")
    if not (os.path.exists(p) and os.path.exists(t)):
        return None
    return np.load(p).squeeze(), np.load(t).squeeze()


def station_grid(run_dir, df_run, station, out_dir, lang="en"):
    t = STRINGS[lang]
    labels = sorted(df_run["label"].unique())
    colors = assign_colors(labels)
    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.8, 2.9 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    run_name = df_run.iloc[0]["run"]
    drawn = 0
    for ax, label in zip(axes.flat, labels):
        data = _load_pred_true(run_dir, df_run, label, station)
        if data is None:
            ax.axis("off")
            continue
        pred, true = data
        marker = dict(marker="o", ms=3) if len(true) < 15 else {}
        x = np.arange(len(true))
        ax.plot(x, true, color=INK2, lw=1.7, label=t["true"], **marker)
        ax.plot(x, pred, color=colors[label], lw=1.7, label=t["pred"], **marker)
        mse = float(np.mean((pred - true) ** 2)); mae = float(np.mean(np.abs(pred - true)))
        _style(ax)
        ax.set_title(f"{label}   (MSE = {mse:.1f}, MAE = {mae:.2f})",
                     fontsize=9.3, color=INK, loc="left")
        ax.legend(fontsize=7.6, frameon=False, loc="best")
        drawn += 1
    for ax in axes.flat[len(labels):]:
        ax.axis("off")
    if not drawn:
        plt.close(fig)
        return None
    for ax in axes[-1]:
        ax.set_xlabel(t["xlabel"], fontsize=9, color=INK2)
    for row in axes:
        row[0].set_ylabel(t["ylabel"], fontsize=9, color=INK2)
    fig.suptitle(t["grid_title"].format(run=run_name, station=station),
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"pred_grid_{run_name}_{station}.png")
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return path


def station_overlay(run_dir, df_run, station, out_dir, lang="en"):
    t = STRINGS[lang]
    labels = sorted(df_run["label"].unique())
    colors = assign_colors(labels)
    run_name = df_run.iloc[0]["run"]
    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    true_drawn = False
    for label in labels:
        data = _load_pred_true(run_dir, df_run, label, station)
        if data is None:
            continue
        pred, true = data
        marker = dict(marker="o", ms=3) if len(true) < 15 else {}
        x = np.arange(len(true))
        if not true_drawn:
            ax.plot(x, true, color=INK, lw=2.3, label=t["true"], zorder=5, **marker)
            true_drawn = True
        ax.plot(x, pred, color=colors[label], lw=1.5, label=label, alpha=0.95, **marker)
    if not true_drawn:
        plt.close(fig)
        return None
    _style(ax)
    ax.set_xlabel(t["xlabel"], fontsize=9, color=INK2)
    ax.set_ylabel(t["ylabel"], fontsize=9, color=INK2)
    ax.set_title(t["overlay_title"].format(run=run_name, station=station),
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=8.3, frameon=False, ncol=min(len(labels) + 1, 5),
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    path = os.path.join(out_dir, f"pred_overlay_{run_name}_{station}.png")
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return path
