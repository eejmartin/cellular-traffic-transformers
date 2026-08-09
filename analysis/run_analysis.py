"""Complete post-training analysis of one or more result_data_* runs.

Parses every model's result.txt / metrics.npy / configs, writes aggregate
tables (CSV), comparison charts, true-vs-predicted figures for representative
stations, optional parameter counts, and a markdown report tying it together.

Model names are discovered from the run folders and labeled via models/*.py
(through MODEL_CLASSES in exp/exp_main.py), so new or renamed models are
picked up without touching this code.

Usage (from the repository root):

    # analyze one run
    python analysis/run_analysis.py result_data_20260705_131934

    # compare two runs side by side, Macedonian labels, custom output folder
    python analysis/run_analysis.py \
        master_thesis/result_data_20260705_131934 \
        master_thesis/result_data_20260706_072702 \
        --names "Set 1 (bs)" "Set 2 (big_data)" --lang mk -o analysis_results

    # explicit stations for the prediction figures, skip parameter counting
    python analysis/run_analysis.py <run_dir> --stations bs_6260 bs_5113 --no-params
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charts
import pred_plots
import report
from metrics_parser import parse_runs, summarize
from params import count_params


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+",
                    help="one or more result_data_* directories")
    ap.add_argument("--names", nargs="*", default=None,
                    help="display names for the runs (default: folder names)")
    ap.add_argument("-o", "--out", default="analysis_results",
                    help="output directory (default: ./analysis_results)")
    ap.add_argument("--lang", choices=["en", "mk"], default="en",
                    help="language of chart labels and the report")
    ap.add_argument("--stations", nargs="*", default=None,
                    help="stations for prediction figures (default: auto best/median/worst per run)")
    ap.add_argument("--n-stations", type=int, default=3,
                    help="number of auto-selected stations per run (default 3)")
    ap.add_argument("--data-root", default="data",
                    help="data folder used to resolve station categories (default ./data)")
    ap.add_argument("--no-params", action="store_true",
                    help="skip parameter counting (avoids importing torch)")
    args = ap.parse_args()

    if args.names and len(args.names) != len(args.run_dirs):
        ap.error("--names must have one name per run directory")

    os.makedirs(args.out, exist_ok=True)

    # 1. parse metrics -------------------------------------------------------
    df = parse_runs(args.run_dirs, args.names, args.data_root)
    df.to_csv(os.path.join(args.out, "metrics_all.csv"), index=False)
    summary = summarize(df)
    summary.to_csv(os.path.join(args.out, "summary_by_model.csv"), index=False)
    print(f"Parsed {len(df)} (model × station) results "
          f"from {df['run'].nunique()} run(s); models: {sorted(df['label'].unique())}")
    print(summary.to_string(index=False))

    # 2. aggregate charts ----------------------------------------------------
    figures = [
        charts.bar_metric(df, "rse", args.out, args.lang),
        charts.bar_metric(df, "mae", args.out, args.lang),
        charts.box_metric(df, "rse", args.out, args.lang),
        charts.win_counts(df, args.out, args.lang),
        charts.group_breakdown(df, args.out, lang=args.lang),
    ]

    # 3. per-station prediction figures --------------------------------------
    run_dirs = {n: d for n, d in zip(list(dict.fromkeys(df["run"])), args.run_dirs)}
    station_map = {}
    for run, run_dir in run_dirs.items():
        df_run = df[df.run == run]
        stations = args.stations or pred_plots.select_stations(df_run, args.n_stations)
        stations = [s for s in stations if s in set(df_run["station"])]
        station_map[run] = stations
        for st in stations:
            figures.append(pred_plots.station_grid(run_dir, df_run, st, args.out, args.lang))
            figures.append(pred_plots.station_overlay(run_dir, df_run, st, args.out, args.lang))

    # 4. parameter counts (optional) -----------------------------------------
    if not args.no_params:
        counts = count_params(args.run_dirs, args.names)
        if counts:
            figures.append(charts.param_chart(counts, args.out, args.lang))

    # 5. report ---------------------------------------------------------------
    rep = report.write_report(df, summary, args.out, figures, station_map, args.lang)
    n_figs = sum(1 for f in figures if f)
    print(f"\nDone: {n_figs} figures, 2 CSV tables and {os.path.basename(rep)} "
          f"written to {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
