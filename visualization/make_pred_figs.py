# -*- coding: utf-8 -*-
"""True-vs-predicted comparison figures from saved pred.npy/true.npy arrays,
plus copies of the original pipeline-produced plots for the appendix."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os, glob, shutil

# --- CONFIG -----------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MT = os.path.join(REPO, "master_thesis")        # runs live here
OUT = os.path.join(MT, "src", "figures")        # output folder
RUNS = {"bs": "result_data_20260705_131934", "big": "result_data_20260706_072702"}
# ----------------------------------------------------------------------
os.makedirs(os.path.join(OUT, "orig"), exist_ok=True)

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; BASE = "#c3c2b7"
MODEL_COLORS = {"TST": "#2a78d6", "CATS": "#1baf7a", "DeformableTST": "#eda100", "PerimidFormer": "#008300"}
MODEL_LABELS = {"TST": "PatchTST", "CATS": "CATS", "DeformableTST": "DeformableTST", "PerimidFormer": "Peri-midFormer"}
MODELS = ["TST", "CATS", "DeformableTST", "PerimidFormer"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

def style_ax(ax):
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8.5)

def load_pred_true(run, model, station_suffix):
    folder = [f for f in glob.glob(f"{MT}/{RUNS[run]}/{model}/results/*_{station_suffix}")
              if f.endswith("_" + station_suffix)]
    assert len(folder) == 1, (run, model, station_suffix, folder)
    p = np.load(os.path.join(folder[0], "pred.npy")).squeeze()
    t = np.load(os.path.join(folder[0], "true.npy")).squeeze()
    return p, t

def compare_fig(run, station_suffix, title, fname, marker=False):
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 5.6), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, MODELS):
        p, t = load_pred_true(run, m, station_suffix)
        x = np.arange(len(t))
        kw = dict(marker="o", ms=3) if marker else {}
        ax.plot(x, t, color=INK2, lw=1.7, label="реални вредности", **kw)
        ax.plot(x, p, color=MODEL_COLORS[m], lw=1.7, label="прогноза", **kw)
        mse = float(np.mean((p - t) ** 2)); mae = float(np.mean(np.abs(p - t)))
        style_ax(ax)
        ax.set_title(f"{MODEL_LABELS[m]}   (MSE = {mse:.1f}, MAE = {mae:.2f})", fontsize=9.3, color=INK, loc="left")
        ax.legend(fontsize=7.6, frameon=False, loc="best")
    for ax in axes[1]: ax.set_xlabel("тест прозорец (час)", fontsize=9, color=INK2)
    for ax in axes[:, 0]: ax.set_ylabel("корисници", fontsize=9, color=INK2)
    fig.suptitle(title, fontsize=11, color=INK, x=0.02, ha="left")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{OUT}/{fname}", dpi=170, bbox_inches="tight"); plt.close()

def overlay_fig(run, station_suffix, title, fname, marker=False):
    fig, ax = plt.subplots(figsize=(9.8, 3.8))
    first = True
    for m in MODELS:
        p, t = load_pred_true(run, m, station_suffix)
        x = np.arange(len(t))
        kw = dict(marker="o", ms=3) if marker else {}
        if first:
            ax.plot(x, t, color=INK, lw=2.3, label="реални вредности", zorder=5, **kw)
            first = False
        ax.plot(x, p, color=MODEL_COLORS[m], lw=1.5, label=MODEL_LABELS[m], alpha=0.95, **kw)
    style_ax(ax)
    ax.set_xlabel("тест прозорец (час)", fontsize=9, color=INK2)
    ax.set_ylabel("корисници", fontsize=9, color=INK2)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=8.3, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    plt.savefig(f"{OUT}/{fname}", dpi=170, bbox_inches="tight"); plt.close()

compare_fig("bs", "bs_6260", "Сет 1, станица bs_6260 (голема): реални наспроти предвидени вредности, тест множество", "fig_pred_bs6260.png")
overlay_fig("bs", "bs_6260", "Сет 1, станица bs_6260: сите четири модели наспроти реалните вредности", "fig_overlay_bs6260.png")
compare_fig("bs", "bs_5113", "Сет 1, станица bs_5113 (средна): реални наспроти предвидени вредности, тест множество", "fig_pred_bs5113.png")
compare_fig("big", "4G_bs_0", "Сет 2 (big_data), станица 4G_bs_0: реални наспроти предвидени вредности", "fig_pred_4g0.png", marker=True)
overlay_fig("big", "5G_bs_12162", "Сет 2 (big_data), станица 5G_bs_12162: сите четири модели наспроти реалните вредности", "fig_overlay_5g.png", marker=True)

# ------------------------------------------------ appendix: original pipeline plots
for m in MODELS:
    for run, pat, tag in [("bs", "*bs_6260.png", "bs"), ("big", "*4G_bs_0.png", "big")]:
        hits = glob.glob(f"{MT}/{RUNS[run]}/{m}/plots/{pat}")
        if hits:
            shutil.copy(hits[0], f"{OUT}/orig/orig_{tag}_{m}.png")

print("pred figs done")
