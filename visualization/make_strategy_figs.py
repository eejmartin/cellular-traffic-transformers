# -*- coding: utf-8 -*-
"""Multi-strategy transfer-learning comparison figures for the thesis (10.12).

Puts all six training regimes (baseline + five transfer strategies) side by
side per model, one panel per dataset; dashed lines: naive mean (red, RSE=1)
and naive persistence (grey, 0.785). Output: master_thesis/src/figures/.
"""
import os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MT = os.path.join(REPO, "master_thesis_final")
OUT = os.path.join(REPO, "master_thesis", "src", "figures")
sys.path.insert(0, os.path.join(REPO, "analysis"))
os.makedirs(OUT, exist_ok=True)

from metrics_parser import parse_run

# --- CONFIG: (dataset -> {strategy label -> run folder}) --------------
RUNS = {
    "bs": {
        "без трансфер": "result_data_bs_s2021_20260809_072721",
        "секвенцијален": "result_data_bs_transfer-sequential_s2021_20260809_201853",
        "ordered": "result_data_bs_transfer-ordered_s2021_20260810_021407",
        "grouped": "result_data_bs_transfer-grouped-g3_s2021_20260810_035824",
        "federated": "result_data_bs_transfer-federated-r5-e5_s2021_20260810_054314",
        "комбиниран": "result_data_bs_transfer-combined-g3-r5-e5_s2021_20260810_081359",
    },
    "big": {
        "без трансфер": "result_data_big_data_l100_s2021_20260809_084717",
        "секвенцијален": "result_data_big_data_l100_transfer-sequential_s2021_20260809_215939",
        "ordered": "result_data_big_data_l100_transfer-ordered_s2021_20260810_033643",
        "grouped": "result_data_big_data_l100_transfer-grouped-g3_s2021_20260810_052200",
        "federated": "result_data_big_data_l100_transfer-federated-r5-e5_s2021_20260810_074112",
        "комбиниран": "result_data_big_data_l100_transfer-combined-g3-r5-e5_s2021_20260810_101031",
    },
}
STRATS = ["без трансфер", "секвенцијален", "ordered", "grouped", "federated", "комбиниран"]
MODELS_ORDER = ["PatchTST", "CATS", "DeformableTST", "PerimidFormer"]
PANELS = [("bs", "Сет 1: 91 станици (bs)"), ("big", "Сет 2: big_data, први 100 станици")]
# ----------------------------------------------------------------------

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; BASE = "#c3c2b7"
# one hue per strategy (baseline muted, transfer modes distinct)
STRAT_COLOR = {"без трансфер": "#c3c2b7", "секвенцијален": "#2a78d6", "ordered": "#1baf7a",
               "grouped": "#eda100", "federated": "#8d67c6", "комбиниран": "#eb6834"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

def style_ax(ax):
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8.5)

def mean_rse(run_dir):
    df = parse_run(os.path.join(MT, run_dir))
    return df.groupby("label")["rse"].mean()

# gather
data = {}   # (dataset, strategy) -> Series(label->rse)
for ds, strat_runs in RUNS.items():
    for strat, folder in strat_runs.items():
        data[(ds, strat)] = mean_rse(folder)

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1))
for ax, (ds, title) in zip(axes, PANELS):
    x = np.arange(len(MODELS_ORDER)); w = 0.14
    for k, strat in enumerate(STRATS):
        series = data[(ds, strat)]
        vals = [series.get(m, np.nan) for m in MODELS_ORDER]
        bars = ax.bar(x + (k - 2.5) * w, vals, w * 0.92, color=STRAT_COLOR[strat],
                      edgecolor="white", lw=0.5, label=strat)
        for r, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.2f}", ha="center",
                        va="bottom", fontsize=6.6, color=INK2, rotation=90)
    ax.axhline(1.0, color="#e34948", lw=0.9, ls=(0, (4, 3)))
    ax.axhline(0.785, color="#8a8880", lw=0.9, ls=(0, (2, 2)))
    ax.set_xticks(x); ax.set_xticklabels(MODELS_ORDER, fontsize=8, rotation=12)
    style_ax(ax); ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.margins(y=0.20)
axes[0].set_ylabel("просечен RSE", fontsize=9, color=INK2)
axes[0].legend(fontsize=8.2, frameon=False, loc="upper left")
fig.tight_layout()
path = os.path.join(OUT, "fig_transfer_strategies.png")
fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
print("wrote", path)

# print summary table for the thesis text
print("\nmean RSE by strategy:")
for ds, title in PANELS:
    print(f"-- {ds} --")
    for m in MODELS_ORDER:
        row = " ".join(f"{s}={data[(ds, s)].get(m, float('nan')):.3f}" for s in STRATS)
        print(f"  {m:15s} {row}")
