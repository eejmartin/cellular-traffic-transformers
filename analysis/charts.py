"""Aggregate comparison charts built from the parsed metrics DataFrame.

All functions take the tidy DataFrame produced by metrics_parser and write a
PNG into ``out_dir``. Figures get one panel per run, so passing several runs
compares them side by side. Model colors are stable across all figures.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from models_registry import assign_colors

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"; RED = "#e34948"

STRINGS = {
    "en": {
        "mean": "mean", "median": "median", "users": "users",
        "rse_axis": "RSE (mean / median)", "mae_axis": "MAE (users)",
        "mse_axis": "MSE (mean / median)",
        "rse_station": "RSE per station",
        "naive": "RSE = 1 (naive mean)",
        "wins_axis": "stations with lowest MSE",
        "group_axis": "mean RSE per group",
        "params_axis": "trainable parameters (thousands)",
    },
    "mk": {
        "mean": "просек", "median": "медијана", "users": "корисници",
        "rse_axis": "RSE (просек / медијана)", "mae_axis": "MAE (корисници)",
        "mse_axis": "MSE (просек / медијана)",
        "rse_station": "RSE по станица",
        "naive": "RSE = 1 (наивна средина)",
        "wins_axis": "станици со најнизок MSE",
        "group_axis": "просечен RSE по група",
        "params_axis": "параметри (илјади)",
    },
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


def _runs_and_labels(df):
    runs = list(dict.fromkeys(df["run"]))
    labels = sorted(df["label"].unique())
    return runs, labels, assign_colors(labels)


def _panels(n_runs):
    fig, axes = plt.subplots(1, n_runs, figsize=(5.2 * n_runs, 3.7), squeeze=False)
    return fig, axes[0]


def _fmt(v):
    return f"{v:,.0f}" if abs(v) >= 1000 else (f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}")


def bar_metric(df, metric, out_dir, lang="en", fname=None):
    """Grouped bars of mean + median of ``metric`` per model, one panel per run."""
    t = STRINGS[lang]
    runs, labels, colors = _runs_and_labels(df)
    fig, axes = _panels(len(runs))
    for ax, run in zip(axes, runs):
        sub = df[df.run == run].groupby("label")[metric].agg(["mean", "median"]).reindex(labels)
        x = np.arange(len(labels)); w = 0.38
        b1 = ax.bar(x - w / 2, sub["mean"], w * 0.92,
                    color=[colors[l] for l in labels], edgecolor="white", lw=0.5)
        b2 = ax.bar(x + w / 2, sub["median"], w * 0.92,
                    color=[colors[l] for l in labels], alpha=0.45, edgecolor="white", lw=0.5)
        for bars, vals, col in ((b1, sub["mean"], INK), (b2, sub["median"], INK2)):
            for r, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(r.get_x() + r.get_width() / 2, v, _fmt(v),
                            ha="center", va="bottom", fontsize=7.6, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.2, rotation=12 if len(labels) > 4 else 0)
        _style(ax)
        ax.set_title(run, fontsize=9.5, color=INK, loc="left")
        ax.margins(y=0.15)
    axes[0].set_ylabel(t.get(f"{metric}_axis", metric.upper()), fontsize=9, color=INK2)
    axes[-1].legend(handles=[Patch(fc=INK2, label=t["mean"]),
                             Patch(fc=INK2, alpha=0.45, label=t["median"])],
                    fontsize=8.5, frameon=False, loc="best")
    fig.tight_layout()
    path = os.path.join(out_dir, fname or f"bar_{metric}.png")
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return path


def box_metric(df, metric, out_dir, lang="en", fname=None):
    """Per-station distribution of ``metric`` per model (boxplots)."""
    t = STRINGS[lang]
    runs, labels, colors = _runs_and_labels(df)
    fig, axes = _panels(len(runs))
    for ax, run in zip(axes, runs):
        data = [df[(df.run == run) & (df.label == l)][metric].dropna().values for l in labels]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        flierprops=dict(marker="o", ms=3, mfc=MUTED, mec="none", alpha=0.6),
                        medianprops=dict(color=INK, lw=1.4),
                        whiskerprops=dict(color=BASE), capprops=dict(color=BASE))
        for patch, l in zip(bp["boxes"], labels):
            patch.set_facecolor(colors[l]); patch.set_alpha(0.65); patch.set_edgecolor("white")
        if metric == "rse":
            ax.axhline(1.0, color=RED, lw=1.1, ls=(0, (4, 3)))
            ax.text(0.99, 1.0, " " + t["naive"], fontsize=7.3, color=RED,
                    va="bottom", ha="right", transform=ax.get_yaxis_transform())
        ax.set_xticklabels(labels, fontsize=8.2, rotation=12 if len(labels) > 4 else 0)
        _style(ax)
        ax.set_title(run, fontsize=9.5, color=INK, loc="left")
    axes[0].set_ylabel(t["rse_station"] if metric == "rse" else metric.upper(),
                       fontsize=9, color=INK2)
    fig.tight_layout()
    path = os.path.join(out_dir, fname or f"box_{metric}.png")
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return path


def win_counts(df, out_dir, lang="en", fname="wins.png"):
    """Number of stations where each model has the lowest MSE."""
    t = STRINGS[lang]
    runs, labels, colors = _runs_and_labels(df)
    fig, axes = _panels(len(runs))
    for ax, run in zip(axes, runs):
        sub = df[df.run == run].pivot_table(index="station", columns="label", values="mse")
        wins = sub.idxmin(axis=1).value_counts().reindex(labels).fillna(0).astype(int)
        bars = ax.bar(np.arange(len(labels)), wins.values, 0.6,
                      color=[colors[l] for l in labels], edgecolor="white", lw=0.5)
        for r, v in zip(bars, wins.values):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.4, str(v),
                    ha="center", fontsize=9, color=INK, fontweight="bold")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.2, rotation=12 if len(labels) > 4 else 0)
        _style(ax)
        ax.set_title(run, fontsize=9.5, color=INK, loc="left")
        ax.margins(y=0.18)
    axes[0].set_ylabel(t["wins_axis"], fontsize=9, color=INK2)
    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return path


def group_breakdown(df, out_dir, metric="rse", lang="en", fname="group_breakdown.png"):
    """Mean of ``metric`` per station group (category / technology) per model.
    Skipped (returns None) when a run has only one group."""
    t = STRINGS[lang]
    runs, labels, colors = _runs_and_labels(df)
    runs = [r for r in runs if df[df.run == r]["group"].nunique() > 1]
    if not runs:
        return None
    fig, axes = _panels(len(runs))
    for ax, run in zip(axes, runs):
        sub = df[df.run == run].groupby(["group", "label"])[metric].mean().unstack()
        sub = sub.reindex(columns=labels)
        groups = list(sub.index)
        x = np.arange(len(groups)); w = 0.8 / max(len(labels), 1)
        for i, l in enumerate(labels):
            vals = sub[l].values
            bars = ax.bar(x + (i - (len(labels) - 1) / 2) * w, vals, w * 0.9,
                          color=colors[l], edgecolor="white", lw=0.5, label=l)
            for r, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(r.get_x() + r.get_width() / 2, v, _fmt(v), ha="center",
                            va="bottom", fontsize=6.4, color=INK2, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
        _style(ax)
        ax.set_title(run, fontsize=9.5, color=INK, loc="left")
        ax.margins(y=0.22)
    axes[0].set_ylabel(t["group_axis"], fontsize=9, color=INK2)
    axes[-1].legend(fontsize=7.8, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return path


def param_chart(param_counts, out_dir, lang="en", fname="params.png"):
    """Bar chart of trainable parameters per model.

    ``param_counts``: {run_name: {label: n_params}} (see params.py).
    """
    t = STRINGS[lang]
    runs = list(param_counts)
    labels = sorted({l for r in runs for l in param_counts[r]})
    colors = assign_colors(labels)
    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 1.5, 3.3))
    w = 0.8 / max(len(runs), 1)
    x = np.arange(len(labels))
    for k, run in enumerate(runs):
        vals = [param_counts[run].get(l, np.nan) / 1000 for l in labels]
        alpha = 1.0 - 0.45 * (k / max(len(runs) - 1, 1)) if len(runs) > 1 else 1.0
        bars = ax.bar(x + (k - (len(runs) - 1) / 2) * w, vals, w * 0.9,
                      color=[colors[l] for l in labels], alpha=alpha,
                      edgecolor="white", lw=0.5)
        for r, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.1f}k",
                        ha="center", va="bottom", fontsize=7.6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    if len(runs) > 1:
        ax.legend(handles=[Patch(fc=INK2, alpha=1.0 - 0.45 * (k / (len(runs) - 1)), label=r)
                           for k, r in enumerate(runs)],
                  fontsize=8, frameon=False)
    _style(ax)
    ax.set_ylabel(t["params_axis"], fontsize=9, color=INK2)
    ax.margins(y=0.15)
    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return path
