# -*- coding: utf-8 -*-
"""Data-series figures and result-comparison charts for the thesis.

Reads the metrics CSV (regenerate it with analysis/run_analysis.py if
needed) and the repo's data/ folder. Parameter counts come from
analysis/params.py. All paths and run IDs are set in the CONFIG block.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys

# --- CONFIG -----------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MT = os.path.join(REPO, "master_thesis")        # thesis assets folder
MTF = os.path.join(REPO, "master_thesis_final")  # v2 campaign runs
OUT = os.path.join(MT, "src", "figures")        # output folder
METRICS_CSV = os.path.join(MT, "metrics_all.csv")
RUNS = {"bs": "result_data_bs_s2021_20260809_072721", "big": "result_data_big_data_l100_s2021_20260809_084717"}
# ----------------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "analysis"))

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; BASE = "#c3c2b7"
MODEL_COLORS = {"TST": "#2a78d6", "CATS": "#1baf7a", "DeformableTST": "#eda100", "PerimidFormer": "#008300"}
MODEL_LABELS = {"TST": "PatchTST", "CATS": "CATS", "DeformableTST": "DeformableTST", "PerimidFormer": "Peri-midFormer"}
MODELS = ["TST", "CATS", "DeformableTST", "PerimidFormer"]
RED = "#e34948"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

def style_ax(ax):
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8.5)

df = pd.read_csv(METRICS_CSV)
# derive 'category': bs -> big/medium/small from the folder group; big -> 4G/5G
def _cat(row):
    g = str(row.get("group", ""))
    if g.endswith("_bs"):
        return g[:-3]
    s = str(row["station"]) + " " + str(row["setting"])
    return "4G" if "4G" in s else "5G"
df["category"] = df.apply(_cat, axis=1)

# ------------------------------------------------ A. daily cycles, one station
d = pd.read_csv(f"{REPO}/data/big_bs/bs_6260.csv", parse_dates=["time_hour"])
fig, ax = plt.subplots(figsize=(9.5, 3.4))
ax.plot(d["time_hour"], d["users"], color=MODEL_COLORS["TST"], lw=1.6)
style_ax(ax)
ax.set_xlabel("време", fontsize=9, color=INK2); ax.set_ylabel("број на активни корисници", fontsize=9, color=INK2)
ax.set_title("Часовен број на корисници, станица bs_6260 (категорија big), 192 часа", fontsize=10.5, color=INK, loc="left")
fig.autofmt_xdate()
plt.tight_layout(); plt.savefig(f"{OUT}/fig_daily_cycles.png", dpi=170, bbox_inches="tight"); plt.close()

# ------------------------------------------------ B. three station categories
examples = [("big_bs", "bs_6260", "голема (big)"), ("medium_bs", "bs_5113", "средна (medium)"), ("small_bs", "bs_2493", "мала (small)")]
avail = {c: sorted(os.listdir(f"{REPO}/data/{c}")) for c, _, _ in examples}
fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.8), sharex=False)
for ax, (cat, st, lab) in zip(axes, examples):
    f = f"{REPO}/data/{cat}/{st}.csv"
    if not os.path.exists(f):
        st = avail[cat][0][:-4]; f = f"{REPO}/data/{cat}/{st}.csv"
    d = pd.read_csv(f, parse_dates=["time_hour"])
    ax.plot(np.arange(len(d)), d["users"], color=MODEL_COLORS["TST"], lw=1.4)
    style_ax(ax)
    ax.set_ylabel("корисници", fontsize=9, color=INK2)
    ax.set_title(f"{lab} базна станица — {st}", fontsize=9.5, color=INK, loc="left")
axes[-1].set_xlabel("време (часови)", fontsize=9, color=INK2)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_categories.png", dpi=170, bbox_inches="tight"); plt.close()

# ------------------------------------------------ C. big_data 48h series (4G + 5G)
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
for ax, (tech, st) in zip(axes, [("4G", "bs_0"), ("5G", "bs_12162")]):
    f = f"{REPO}/data/big_data/parsed/concat/{tech}/{st}.csv"
    if not os.path.exists(f):
        st = sorted(os.listdir(f"{REPO}/data/big_data/parsed/concat/{tech}"))[0][:-4]
        f = f"{REPO}/data/big_data/parsed/concat/{tech}/{st}.csv"
    d = pd.read_csv(f, parse_dates=["time_hour"])
    ax.plot(np.arange(len(d)), d["users"], color=MODEL_COLORS["TST"], lw=1.7, marker="o", ms=2.6)
    ax.axvline(23.5, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(11.5, ax.get_ylim()[1], "работен ден", fontsize=8.3, color=MUTED, ha="center", va="top")
    ax.text(35.5, ax.get_ylim()[1], "викенд", fontsize=8.3, color=MUTED, ha="center", va="top")
    style_ax(ax)
    ax.set_title(f"{tech} станица {st} (48 ч.)", fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("време (часови)", fontsize=9, color=INK2)
axes[0].set_ylabel("корисници", fontsize=9, color=INK2)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_bigdata_series.png", dpi=170, bbox_inches="tight"); plt.close()

# ------------------------------------------------ D. mean/median bars (both runs)
def grouped_bar(metric, fname, title, fmt="{:.2f}"):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
    for ax, run, rt in [(axes[0], "bs", "Сет 1: 91 станици (bs)"), (axes[1], "big", "Сет 2: big_data, први 100 станици")]:
        sub = df[df.run == run].groupby("model")[metric].agg(["mean", "median"]).reindex(MODELS)
        x = np.arange(len(MODELS)); w = 0.38
        b1 = ax.bar(x - w/2, sub["mean"].values, w * 0.92, color=[MODEL_COLORS[m] for m in MODELS], edgecolor="white", lw=0.5)
        b2 = ax.bar(x + w/2, sub["median"].values, w * 0.92, color=[MODEL_COLORS[m] for m in MODELS], alpha=0.45, edgecolor="white", lw=0.5)
        for r, v in zip(b1, sub["mean"].values):
            ax.text(r.get_x() + r.get_width()/2, v, fmt.format(v), ha="center", va="bottom", fontsize=7.8, color=INK)
        for r, v in zip(b2, sub["median"].values):
            ax.text(r.get_x() + r.get_width()/2, v, fmt.format(v), ha="center", va="bottom", fontsize=7.8, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=8.2)
        style_ax(ax); ax.set_title(rt, fontsize=9.5, color=INK, loc="left")
        ax.margins(y=0.15)
    axes[0].set_ylabel(title, fontsize=9, color=INK2)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(fc=INK2, label="просек"), Patch(fc=INK2, alpha=0.45, label="медијана")],
                   fontsize=8.5, frameon=False, loc="upper left")
    plt.tight_layout(); plt.savefig(f"{OUT}/{fname}", dpi=170, bbox_inches="tight"); plt.close()

grouped_bar("rse", "fig_rse_bar.png", "RSE (просек / медијана)")
grouped_bar("mae", "fig_mae_bar.png", "MAE (број на корисници)", fmt="{:.1f}")

# ------------------------------------------------ E. RSE boxplots
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
for ax, run, rt in [(axes[0], "bs", "Сет 1: 91 станици (bs)"), (axes[1], "big", "Сет 2: big_data, први 100 станици")]:
    data = [df[(df.run == run) & (df.model == m)]["rse"].values for m in MODELS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                    flierprops=dict(marker="o", ms=3, mfc=MUTED, mec="none", alpha=0.6),
                    medianprops=dict(color=INK, lw=1.4),
                    whiskerprops=dict(color=BASE), capprops=dict(color=BASE))
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(MODEL_COLORS[m]); patch.set_alpha(0.65); patch.set_edgecolor("white")
    ax.axhline(1.0, color=RED, lw=1.1, ls=(0, (4, 3)))
    ax.text(4.45, 1.0, "RSE = 1\n(наивна\nсредина)", fontsize=7.5, color=RED, va="center")
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=8.2)
    style_ax(ax); ax.set_title(rt, fontsize=9.5, color=INK, loc="left")
axes[0].set_ylabel("RSE по станица", fontsize=9, color=INK2)
axes[0].set_ylim(0, 2.2); axes[1].set_ylim(0, 6.5)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_rse_box.png", dpi=170, bbox_inches="tight"); plt.close()

# ------------------------------------------------ F. win counts
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.3))
for ax, run, rt in [(axes[0], "bs", "Сет 1: 91 станици (bs)"), (axes[1], "big", "Сет 2: big_data, први 100 станици")]:
    sub = df[df.run == run].pivot(index="station", columns="model", values="mse")
    wins = sub.idxmin(axis=1).value_counts().reindex(MODELS).fillna(0).astype(int)
    bars = ax.bar(np.arange(len(MODELS)), wins.values, 0.6,
                  color=[MODEL_COLORS[m] for m in MODELS], edgecolor="white", lw=0.5)
    for r, v in zip(bars, wins.values):
        ax.text(r.get_x() + r.get_width()/2, v + 0.5, str(v), ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(np.arange(len(MODELS))); ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=8.2)
    style_ax(ax); ax.set_title(rt, fontsize=9.5, color=INK, loc="left")
    ax.margins(y=0.18)
axes[0].set_ylabel("број на станици со најнизок MSE", fontsize=9, color=INK2)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_wins.png", dpi=170, bbox_inches="tight"); plt.close()

# ------------------------------------------------ G. RSE by category (bs) and tech (big)
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
cats = ["big_bs", "medium_bs", "small_bs"]; cat_lab = ["голема", "средна", "мала"]
sub = df[df.run == "bs"].groupby(["category", "model"])["rse"].mean().unstack().reindex(cats)[MODELS]
x = np.arange(len(cats)); w = 0.19
for i, m in enumerate(MODELS):
    bars = axes[0].bar(x + (i - 1.5) * w, sub[m].values, w * 0.9, color=MODEL_COLORS[m],
                       edgecolor="white", lw=0.5, label=MODEL_LABELS[m])
    for r, v in zip(bars, sub[m].values):
        axes[0].text(r.get_x() + r.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.6, color=INK2, rotation=90, va="bottom")
axes[0].set_xticks(x); axes[0].set_xticklabels(cat_lab, fontsize=9)
axes[0].set_title("Сет 1: просечен RSE по категорија на станица", fontsize=9.5, color=INK, loc="left")
axes[0].set_ylabel("просечен RSE", fontsize=9, color=INK2); axes[0].margins(y=0.22)
techs = ["4G", "5G"]
sub2 = df[df.run == "big"].groupby(["category", "model"])["rse"].mean().unstack().reindex(techs)[MODELS]
x2 = np.arange(len(techs))
for i, m in enumerate(MODELS):
    bars = axes[1].bar(x2 + (i - 1.5) * w, sub2[m].values, w * 0.9, color=MODEL_COLORS[m],
                       edgecolor="white", lw=0.5, label=MODEL_LABELS[m])
    for r, v in zip(bars, sub2[m].values):
        axes[1].text(r.get_x() + r.get_width()/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.6, color=INK2, rotation=90, va="bottom")
axes[1].set_xticks(x2); axes[1].set_xticklabels(techs, fontsize=9)
axes[1].set_title("Сет 2: просечен RSE по технологија", fontsize=9.5, color=INK, loc="left")
axes[1].margins(y=0.22)
for ax in axes: style_ax(ax)
axes[1].legend(fontsize=7.8, frameon=False, ncol=2, loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_rse_breakdown.png", dpi=170, bbox_inches="tight"); plt.close()

# ------------------------------------------------ H. parameter counts (via analysis/params.py)
from params import count_params
counts = count_params([os.path.join(MTF, RUNS["bs"]), os.path.join(MTF, RUNS["big"])],
                      run_names=["bs", "big"])
if counts:
    from models_registry import label_for
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    x = np.arange(len(MODELS)); w = 0.38
    v1 = [counts["bs"].get(label_for(m), np.nan) / 1000 for m in MODELS]
    v2 = [counts["big"].get(label_for(m), np.nan) / 1000 for m in MODELS]
    b1 = ax.bar(x - w/2, v1, w * 0.92, color=[MODEL_COLORS[m] for m in MODELS], edgecolor="white", lw=0.5)
    b2 = ax.bar(x + w/2, v2, w * 0.92, color=[MODEL_COLORS[m] for m in MODELS], alpha=0.45, edgecolor="white", lw=0.5)
    for r, v in zip(b1, v1): ax.text(r.get_x() + r.get_width()/2, v, f"{v:.1f}k", ha="center", va="bottom", fontsize=8, color=INK)
    for r, v in zip(b2, v2): ax.text(r.get_x() + r.get_width()/2, v, f"{v:.1f}k", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=8.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=INK2, label="seq_len = 24 (Сет 1)"), Patch(fc=INK2, alpha=0.45, label="seq_len = 12 (Сет 2)")],
              fontsize=8.5, frameon=False)
    style_ax(ax); ax.set_ylabel("број на параметри (илјади)", fontsize=9, color=INK2)
    ax.margins(y=0.15)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_params.png", dpi=170, bbox_inches="tight"); plt.close()

print("charts done")
