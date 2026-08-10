"""Cell-sleep simulation on saved forecasts (§12.11 case study).

Threshold policy evaluated per station on the test predictions of a finished
campaign: "if the forecast for the next hour is below the station's sleep
threshold, put the secondary carrier to sleep". The threshold is the q-th
percentile (default 20 %) of the station's TRAINING-split load — no test
information leaks into the policy.

Per (model, station), compared against the same policy applied to the true
values:
    sleep_hours      hours the forecast-driven policy sleeps (% of test)
    oracle_hours     hours the true-value policy sleeps
    false_sleeps     forecast says sleep, reality is above the threshold
                     (QoS risk — the expensive error)
    missed_sleeps    reality below threshold, forecast above (lost saving)

Outputs <run_dir>_sleepsim.csv + per-model summary.

Usage:
    python analysis/sleep_sim.py <run_dir> [<run_dir> ...] [--quantile 0.2]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_RATIO = 0.7


def station_threshold(cfg, quantile):
    csv = os.path.join(cfg['root_path'], cfg['data_path'])
    users = pd.read_csv(csv)['users'].values.astype(float)
    train = users[:int(len(users) * TRAIN_RATIO)]
    return float(np.quantile(train, quantile))


def process_run(run_dir, quantile):
    rows = []
    for model_dir in sorted(glob.glob(os.path.join(run_dir, '*', 'configs'))):
        model_dir = os.path.dirname(model_dir)
        model = os.path.basename(model_dir)
        for cfg_path in sorted(glob.glob(os.path.join(model_dir, 'configs', '*.json'))):
            with open(cfg_path) as f:
                cfg = json.load(f)
            res = os.path.join(model_dir, 'results', cfg['setting'])
            p_pred, p_true = os.path.join(res, 'pred.npy'), os.path.join(res, 'true.npy')
            if not (os.path.exists(p_pred) and os.path.exists(p_true)):
                continue
            try:
                thr = station_threshold(cfg, quantile)
            except Exception:
                continue
            pred = np.load(p_pred).reshape(-1)
            true = np.load(p_true).reshape(-1)
            sleep = pred < thr
            oracle = true < thr
            n = len(true)
            rows.append({
                'model': model, 'station': os.path.basename(cfg['data_path']),
                'threshold': thr, 'n_hours': n,
                'sleep_share': float(sleep.mean()),
                'oracle_share': float(oracle.mean()),
                'false_sleep_share': float((sleep & ~oracle).mean()),
                'missed_sleep_share': float((~sleep & oracle).mean()),
            })
    return pd.DataFrame(rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('run_dirs', nargs='+')
    cli.add_argument('--quantile', type=float, default=0.2,
                     help='sleep threshold = this quantile of the train load')
    args = cli.parse_args()

    for rd in args.run_dirs:
        rd = rd.rstrip('/')
        print(f'== {os.path.basename(rd)} (threshold: {args.quantile:.0%} of train load)')
        df = process_run(rd, args.quantile)
        if df.empty:
            print('  nothing parsed')
            continue
        out = rd + '_sleepsim.csv'
        df.to_csv(out, index=False)
        summary = df.groupby('model').agg(
            stations=('station', 'size'),
            sleep=('sleep_share', 'mean'),
            oracle=('oracle_share', 'mean'),
            false_sleep=('false_sleep_share', 'mean'),
            missed=('missed_sleep_share', 'mean')).round(4)
        print(summary.to_string())
        print('  ->', out)


if __name__ == '__main__':
    main()
