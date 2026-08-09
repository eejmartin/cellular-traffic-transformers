"""Quantify metric drift between a campaign and its re-evaluation.

After scripts/retest.py produced <run_dir>_retest, this tool compares the
old (truncated test set) metrics against the corrected ones, per model:
mean MSE/MAE/RSE before and after, mean RSE delta, share of stations whose
RSE moved by more than a threshold, and the largest movers. Used to gauge
how much the drop_last evaluation bug distorted previously reported tables.

Usage:
    python analysis/retest_drift.py <run_dir> [<run_dir> ...]
        [--threshold 0.05] [--top 5] [-o output.csv]

Each <run_dir> must have a sibling <run_dir>_retest.
"""

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# setting may contain spaces (DeformableTST encodes stage_spec lists), so
# match the whole line, not \S+
RESULT_RE = re.compile(
    r'^(?P<setting>\S.*?)\s*$\n^mse:(?P<mse>[\d.eE+-]+), mae:(?P<mae>[\d.eE+-]+), '
    r'rse:(?P<rse>[\d.eE+-nan]+)\s*$', re.M)


def parse_result_txt(path):
    """result.txt -> DataFrame(setting, mse, mae, rse). Later duplicates win."""
    with open(path) as f:
        text = f.read()
    rows = {}
    for m in RESULT_RE.finditer(text):
        rows[m.group('setting')] = {
            'setting': m.group('setting'),
            'mse': float(m.group('mse')),
            'mae': float(m.group('mae')),
            'rse': float(m.group('rse')),
        }
    return pd.DataFrame(list(rows.values()))


def collect(run_dir):
    """All (model, setting, metrics) rows of one run folder."""
    frames = []
    for model in sorted(os.listdir(run_dir)):
        result_txt = os.path.join(run_dir, model, 'result.txt')
        if not os.path.exists(result_txt):
            continue
        df = parse_result_txt(result_txt)
        if not df.empty:
            df.insert(0, 'model', model)
            frames.append(df)
    if not frames:
        raise SystemExit(f'no result.txt found under {run_dir}')
    return pd.concat(frames, ignore_index=True)


def drift_report(run_dir, threshold, top):
    old = collect(run_dir)
    new = collect(run_dir + '_retest')
    merged = old.merge(new, on=['model', 'setting'], suffixes=('_old', '_new'))
    merged['rse_delta'] = merged['rse_new'] - merged['rse_old']

    print(f'\n=== {os.path.basename(run_dir)} '
          f'({len(merged)} paired settings) ===')
    summary = merged.groupby('model').agg(
        n=('setting', 'size'),
        mse_old=('mse_old', 'mean'), mse_new=('mse_new', 'mean'),
        mae_old=('mae_old', 'mean'), mae_new=('mae_new', 'mean'),
        rse_old=('rse_old', 'mean'), rse_new=('rse_new', 'mean'),
        rse_delta=('rse_delta', 'mean'),
        moved=('rse_delta', lambda d: int((d.abs() > threshold).sum())),
    )
    with pd.option_context('display.width', 200, 'display.max_columns', 20,
                           'display.float_format', lambda v: f'{v:.4f}'):
        print(summary)

    movers = merged.reindex(merged['rse_delta'].abs()
                            .sort_values(ascending=False).index).head(top)
    print(f'top {top} movers (|ΔRSE|):')
    for _, r in movers.iterrows():
        print(f"  {r['model']:>14}  {r['setting'][-30:]:>30}  "
              f"rse {r['rse_old']:.3f} -> {r['rse_new']:.3f}  "
              f"(Δ {r['rse_delta']:+.3f})")
    return merged.assign(run=os.path.basename(run_dir))


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('run_dirs', nargs='+')
    cli.add_argument('--threshold', type=float, default=0.05,
                     help='|ΔRSE| above this counts as a moved station')
    cli.add_argument('--top', type=int, default=5, help='movers to list per run')
    cli.add_argument('-o', '--output', default=None,
                     help='optional CSV with every paired row of every run')
    args = cli.parse_args()

    all_rows = [drift_report(rd.rstrip('/'), args.threshold, args.top)
                for rd in args.run_dirs]
    if args.output:
        pd.concat(all_rows, ignore_index=True).to_csv(args.output, index=False)
        print(f'\nsaved: {args.output}')


if __name__ == '__main__':
    main()
