"""Naive / seasonal-naive / linear baselines on the exact model test windows.

Computes, per station and dataset, three reference forecasts evaluated on the
same test targets the transformers see (identical chronological 70/10/20 split
and window arithmetic as data_provider/data_loader.py):

    naive       y_hat(t+1) = y(t)                (persistence)
    seasonal    y_hat(t+1) = y(t+1-24)           (same hour yesterday)
    linear      least-squares regression y(t+1) ~ last seq_len values of y,
                fit per station on the training windows only

Outputs under master_thesis_final/baselines_<ts>/:
    baseline_metrics.csv   one row per (set, station, baseline): mse/mae/rse...
    baseline_summary.csv   mean/median per (set, baseline)

CPU-only, no torch. Usage: python scripts/run_baselines.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tools import read_file_names_by_order, read_big_data_file_names  # noqa: E402
from utils.metrics import metric_all  # noqa: E402

SETS = {
    'Set1': dict(lister=read_file_names_by_order, seq_len=24),
    'Set2': dict(lister=lambda: read_big_data_file_names()[:100], seq_len=12),
}
RATIOS = (0.7, 0.1, 0.2)
SEASON = 24            # daily cycle in hours, both sets are hourly
PRED_LEN = 1


def split_borders(n, seq_len):
    """Replicates Dataset_Custom border arithmetic exactly."""
    num_train = int(n * RATIOS[0])
    num_test = int(n * RATIOS[2])
    border1s = [0, num_train - seq_len, n - num_test - seq_len]
    border2s = [num_train, int(n * RATIOS[0]) + (n - num_train - num_test), n]
    return border1s, border2s


def windows(series, lo, hi, seq_len):
    """(X, y, target_idx) for every window fully inside series[lo:hi]."""
    X, y, idx = [], [], []
    for s in range(lo, hi - seq_len - PRED_LEN + 1):
        X.append(series[s:s + seq_len])
        y.append(series[s + seq_len])
        idx.append(s + seq_len)
    return np.asarray(X), np.asarray(y), np.asarray(idx)


def station_baselines(csv_path, seq_len):
    df = pd.read_csv(csv_path)
    series = df['users'].values.astype(float)
    n = len(series)
    b1s, b2s = split_borders(n, seq_len)

    Xtr, ytr, _ = windows(series, b1s[0], b2s[0], seq_len)
    Xte, yte, idx = windows(series, b1s[2], b2s[2], seq_len)
    if len(yte) == 0 or len(ytr) == 0:
        return None

    preds = {'naive': Xte[:, -1]}
    # seasonal lag reaches back into known past (validation/train region)
    preds['seasonal'] = np.array([series[i - SEASON] if i - SEASON >= 0
                                  else series[max(i - 1, 0)] for i in idx])
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    preds['linear'] = np.hstack([Xte, np.ones((len(Xte), 1))]) @ coef

    out = {}
    for name, p in preds.items():
        out[name] = metric_all(p.reshape(-1, 1, 1), yte.reshape(-1, 1, 1))
        out[name]['n_windows'] = len(yte)
    return out


def main():
    out_dir = os.path.join('master_thesis_final',
                           'baselines_' + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'command.txt'), 'w') as f:
        f.write('python ' + ' '.join(sys.argv) + '\n')

    rows = []
    for set_name, cfg in SETS.items():
        files = cfg['lister']()
        print(f'{set_name}: {len(files)} stations')
        for fp in files:
            res = station_baselines(
                os.path.join(fp['root_path'], fp['data_path']), cfg['seq_len'])
            if res is None:
                print('  skipped (too short):', fp['file_name'])
                continue
            for baseline, m in res.items():
                rows.append({'set': set_name, 'station': fp['file_name'],
                             'baseline': baseline, **m})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'baseline_metrics.csv'), index=False)
    summary = (df.groupby(['set', 'baseline'])
                 .agg(n=('station', 'size'),
                      mse_mean=('mse', 'mean'), mae_mean=('mae', 'mean'),
                      rse_mean=('rse', 'mean'), rse_median=('rse', 'median'),
                      smape_mean=('smape', 'mean'), mase_mean=('mase', 'mean'))
                 .round(4))
    summary.to_csv(os.path.join(out_dir, 'baseline_summary.csv'))
    print(summary.to_string())
    print('saved ->', out_dir)


if __name__ == '__main__':
    main()
