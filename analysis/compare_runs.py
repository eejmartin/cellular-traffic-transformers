"""Paired comparison of two runs over the same stations and models.

Typical use: quantify the effect of one changed setting between two runs —
e.g. transfer learning on vs off, a different seq_len, a preprocessing change —
while data and models are identical. Stations are matched by name, so both
runs must cover the same station files.

Outputs (into the chosen output directory):
    compare_detail.csv    per model x station: metric in both runs + delta
    compare_summary.csv   per model: means/medians in both runs, share improved
    cmp_bar_rse.png       mean/median RSE per model, both runs side by side
    cmp_improved.png      share of stations improved by the variant, per model
    cmp_scatter_rse.png   per-station RSE: baseline vs variant, one panel per model
    cmp_delta_box.png     distribution of per-station RSE change, per model
    compare_report.md     tables + auto-written findings

Usage (from the repository root):

    python analysis/compare_runs.py <baseline_run_dir> <variant_run_dir> \
        --names baseline transfer -o analysis_results/transfer_bs --lang mk
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from charts import BASE, GRID, INK, INK2, MUTED, RED, _style
from metrics_parser import parse_run
from models_registry import assign_colors

STRINGS = {
    "en": {
        "rse_axis": "RSE (mean / median)",
        "improved_axis": "stations improved (%)",
        "improved_note": "share of stations where the variant lowers MSE",
        "scatter_x": "RSE — {base}", "scatter_y": "RSE — {var}",
        "delta_axis": "ΔRSE ({var} − {base})",
        "mean": "mean", "median": "median",
        "title": "# Run comparison: {base} vs {var}\n",
        "summary": "## Summary per model\n",
        "cols": ["model", "stations", "RSE mean {base}", "RSE mean {var}",
                 "Δ RSE mean", "RSE median {base}", "RSE median {var}",
                 "MAE mean {base}", "MAE mean {var}", "improved (%)"],
        "findings": "## Findings\n",
        "verdict_better": "- **{model}**: the variant improves mean RSE by {gain:.3f} "
                          "({base_v:.3f} → {var_v:.3f}); {imp:.0f}% of stations improved.",
        "verdict_worse": "- **{model}**: the variant worsens mean RSE by {gain:.3f} "
                         "({base_v:.3f} → {var_v:.3f}); only {imp:.0f}% of stations improved.",
        "verdict_neutral": "- **{model}**: no practical difference in mean RSE "
                           "({base_v:.3f} → {var_v:.3f}); {imp:.0f}% of stations improved.",
        "figures": "## Figures\n",
    },
    "mk": {
        "rse_axis": "RSE (просек / медијана)",
        "improved_axis": "подобрени станици (%)",
        "improved_note": "удел на станици каде варијантата го намалува MSE",
        "scatter_x": "RSE — {base}", "scatter_y": "RSE — {var}",
        "delta_axis": "ΔRSE ({var} − {base})",
        "mean": "просек", "median": "медијана",
        "title": "# Споредба на извршувања: {base} наспроти {var}\n",
        "summary": "## Резиме по модел\n",
        "cols": ["модел", "станици", "RSE просек {base}", "RSE просек {var}",
                 "Δ RSE просек", "RSE медијана {base}", "RSE медијана {var}",
                 "MAE просек {base}", "MAE просек {var}", "подобрени (%)"],
        "findings": "## Наоди\n",
        "verdict_better": "- **{model}**: варијантата го подобрува просечниот RSE за {gain:.3f} "
                          "({base_v:.3f} → {var_v:.3f}); подобрени {imp:.0f}% од станиците.",
        "verdict_worse": "- **{model}**: варијантата го влошува просечниот RSE за {gain:.3f} "
                         "({base_v:.3f} → {var_v:.3f}); подобрени само {imp:.0f}% од станиците.",
        "verdict_neutral": "- **{model}**: без практична разлика во просечниот RSE "
                           "({base_v:.3f} → {var_v:.3f}); подобрени {imp:.0f}% од станиците.",
        "figures": "## Слики\n",
    },
}

# |mean-RSE change| below this threshold counts as "no practical difference"
NEUTRAL_DELTA = 0.01


def pair_runs(base_dir, var_dir, base_name, var_name, data_root=None):
    """Match (model x station) rows of the two runs into one paired frame."""
    base = parse_run(base_dir, base_name, data_root)
    var = parse_run(var_dir, var_name, data_root)
    merged = base.merge(var, on=["model", "label", "station"],
                        suffixes=("_base", "_var"))
    if merged.empty:
        raise ValueError("The two runs share no (model, station) pairs — "
                         "are they from the same dataset?")
    dropped_base = len(base) - len(merged)
    dropped_var = len(var) - len(merged)
    if dropped_base or dropped_var:
        print(f"pair_runs: dropped {dropped_base} baseline / {dropped_var} "
              "variant rows without a counterpart")
    for metric in ("mse", "mae", "rse"):
        merged[f"delta_{metric}"] = merged[f"{metric}_var"] - merged[f"{metric}_base"]
    merged["improved"] = merged["delta_mse"] < 0
    return merged


def summarize_pairs(paired):
    agg = paired.groupby("label").agg(
        stations=("station", "nunique"),
        rse_mean_base=("rse_base", "mean"), rse_mean_var=("rse_var", "mean"),
        rse_median_base=("rse_base", "median"), rse_median_var=("rse_var", "median"),
        mae_mean_base=("mae_base", "mean"), mae_mean_var=("mae_var", "mean"),
        improved=("improved", "mean"),
    ).reset_index()
    agg["delta_rse_mean"] = agg["rse_mean_var"] - agg["rse_mean_base"]
    return agg.sort_values("label").reset_index(drop=True)


def _bar_pair(summary, base_name, var_name, out_dir, lang):
    t = STRINGS[lang]
    labels = list(summary["label"])
    colors = assign_colors(labels)
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 2.2, 3.6))
    b1 = ax.bar(x - w / 2, summary["rse_mean_base"], w * 0.92,
                color=[colors[l] for l in labels], edgecolor="white", lw=0.5)
    b2 = ax.bar(x + w / 2, summary["rse_mean_var"], w * 0.92,
                color=[colors[l] for l in labels], alpha=0.45,
                edgecolor="white", lw=0.5, hatch="//")
    for bars, vals, col in ((b1, summary["rse_mean_base"], INK),
                            (b2, summary["rse_mean_var"], INK2)):
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.8, color=col)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.4)
    ax.legend(handles=[Patch(fc=INK2, label=base_name),
                       Patch(fc=INK2, alpha=0.45, hatch="//", label=var_name)],
              fontsize=8.5, frameon=False)
    _style(ax)
    ax.set_ylabel(t["rse_axis"], fontsize=9, color=INK2)
    ax.margins(y=0.15)
    fig.tight_layout()
    path = os.path.join(out_dir, "cmp_bar_rse.png")
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return path


def _bar_improved(summary, out_dir, lang):
    t = STRINGS[lang]
    labels = list(summary["label"])
    colors = assign_colors(labels)
    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 1.5, 3.3))
    vals = summary["improved"] * 100
    bars = ax.bar(np.arange(len(labels)), vals, 0.6,
                  color=[colors[l] for l in labels], edgecolor="white", lw=0.5)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 1, f"{v:.0f}%",
                ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.axhline(50, color=RED, lw=1.1, ls=(0, (4, 3)))
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, fontsize=8.4)
    _style(ax)
    ax.set_ylabel(t["improved_axis"], fontsize=9, color=INK2)
    ax.set_ylim(0, max(100, vals.max() + 8))
    ax.set_title(t["improved_note"], fontsize=9, color=INK2, loc="left")
    fig.tight_layout()
    path = os.path.join(out_dir, "cmp_improved.png")
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return path


def _scatter(paired, base_name, var_name, out_dir, lang):
    t = STRINGS[lang]
    labels = sorted(paired["label"].unique())
    colors = assign_colors(labels)
    ncols = 2
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.4, 4.4 * nrows / 1.6),
                             squeeze=False)
    lim = np.nanpercentile(
        np.concatenate([paired["rse_base"], paired["rse_var"]]), 98) * 1.05
    for ax, label in zip(axes.flat, labels):
        sub = paired[paired.label == label]
        ax.plot([0, lim], [0, lim], color=MUTED, lw=1.0, ls=(0, (4, 3)))
        ax.scatter(sub["rse_base"], sub["rse_var"], s=16, color=colors[label],
                   alpha=0.65, edgecolors="none")
        _style(ax)
        imp = 100 * (sub["rse_var"] < sub["rse_base"]).mean()
        ax.set_title(f"{label}  ({imp:.0f}% ↓)", fontsize=9.3, color=INK, loc="left")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    for ax in axes.flat[len(labels):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(t["scatter_x"].format(base=base_name), fontsize=9, color=INK2)
    for row in axes:
        row[0].set_ylabel(t["scatter_y"].format(var=var_name), fontsize=9, color=INK2)
    fig.tight_layout()
    path = os.path.join(out_dir, "cmp_scatter_rse.png")
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return path


def _delta_box(paired, base_name, var_name, out_dir, lang):
    t = STRINGS[lang]
    labels = sorted(paired["label"].unique())
    colors = assign_colors(labels)
    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 1.5, 3.6))
    data = [paired[paired.label == l]["delta_rse"].dropna().values for l in labels]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    flierprops=dict(marker="o", ms=3, mfc=MUTED, mec="none", alpha=0.6),
                    medianprops=dict(color=INK, lw=1.4),
                    whiskerprops=dict(color=BASE), capprops=dict(color=BASE))
    for patch, l in zip(bp["boxes"], labels):
        patch.set_facecolor(colors[l]); patch.set_alpha(0.65); patch.set_edgecolor("white")
    ax.axhline(0, color=RED, lw=1.1, ls=(0, (4, 3)))
    ax.set_xticklabels(labels, fontsize=8.4)
    _style(ax)
    ax.set_ylabel(t["delta_axis"].format(base=base_name, var=var_name),
                  fontsize=9, color=INK2)
    fig.tight_layout()
    path = os.path.join(out_dir, "cmp_delta_box.png")
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return path


def _report(summary, base_name, var_name, out_dir, figures, lang):
    t = STRINGS[lang]
    fmt = dict(base=base_name, var=var_name)
    lines = [t["title"].format(**fmt), t["summary"]]
    header = [c.format(**fmt) for c in t["cols"]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for r in summary.itertuples():
        lines.append("| " + " | ".join([
            r.label, str(r.stations),
            f"{r.rse_mean_base:.3f}", f"{r.rse_mean_var:.3f}",
            f"{r.delta_rse_mean:+.3f}",
            f"{r.rse_median_base:.3f}", f"{r.rse_median_var:.3f}",
            f"{r.mae_mean_base:.2f}", f"{r.mae_mean_var:.2f}",
            f"{100 * r.improved:.0f}",
        ]) + " |")
    lines.append("")
    lines.append(t["findings"])
    for r in summary.itertuples():
        if r.delta_rse_mean <= -NEUTRAL_DELTA:
            key = "verdict_better"
        elif r.delta_rse_mean >= NEUTRAL_DELTA:
            key = "verdict_worse"
        else:
            key = "verdict_neutral"
        lines.append(t[key].format(model=r.label, gain=abs(r.delta_rse_mean),
                                   base_v=r.rse_mean_base, var_v=r.rse_mean_var,
                                   imp=100 * r.improved))
    lines.append("")
    lines.append(t["figures"])
    for p in figures:
        if p:
            lines.append(f"![{os.path.basename(p)}]({os.path.basename(p)})\n")
    path = os.path.join(out_dir, "compare_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def compare(base_dir, var_dir, base_name=None, var_name=None, out_dir="analysis_results",
            lang="en", data_root=None):
    """Full comparison pipeline; returns (paired_detail_df, summary_df)."""
    base_name = base_name or os.path.basename(os.path.abspath(base_dir).rstrip("/"))
    var_name = var_name or os.path.basename(os.path.abspath(var_dir).rstrip("/"))
    os.makedirs(out_dir, exist_ok=True)

    paired = pair_runs(base_dir, var_dir, base_name, var_name, data_root)
    summary = summarize_pairs(paired)
    paired.to_csv(os.path.join(out_dir, "compare_detail.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "compare_summary.csv"), index=False)

    figures = [
        _bar_pair(summary, base_name, var_name, out_dir, lang),
        _bar_improved(summary, out_dir, lang),
        _scatter(paired, base_name, var_name, out_dir, lang),
        _delta_box(paired, base_name, var_name, out_dir, lang),
    ]
    rep = _report(summary, base_name, var_name, out_dir, figures, lang)
    print(summary.to_string(index=False))
    print(f"\nDone: 4 figures, 2 CSV tables and {os.path.basename(rep)} "
          f"written to {os.path.abspath(out_dir)}/")
    return paired, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", help="baseline run directory")
    ap.add_argument("variant", help="variant run directory (same stations & models)")
    ap.add_argument("--names", nargs=2, default=None,
                    help="display names: baseline variant")
    ap.add_argument("-o", "--out", default="analysis_results",
                    help="output directory (default: ./analysis_results)")
    ap.add_argument("--lang", choices=["en", "mk"], default="en")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    names = args.names or (None, None)
    compare(args.baseline, args.variant, names[0], names[1], args.out,
            args.lang, args.data_root)


if __name__ == "__main__":
    main()
