"""Quantify the weekday->weekend concatenation seam in Set 2 (Прилог А).

The 48-hour big_data series glue a representative weekday to a weekend day
at midnight. This tool measures whether that synthetic transition is
statistically unusual: the absolute hour-to-hour change AT the seam
(hour 23 -> 24) versus all other hour-to-hour changes, plus how many sliding
windows cross the seam.

Usage: python analysis/seam_check.py [--limit 100]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tools import read_big_data_file_names  # noqa: E402

SEAM = 24          # row index where the weekend day starts
SEQ_LEN = 12
PRED_LEN = 1
TRAIN_RATIO = 0.7


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--limit', type=int, default=100,
                     help='stations to analyze (default: the thesis first 100)')
    args = cli.parse_args()

    files = read_big_data_file_names()[:args.limit]
    seam_deltas, other_deltas, seam_pctiles = [], [], []
    for fp in files:
        users = pd.read_csv(os.path.join(fp['root_path'], fp['data_path']))['users'] \
            .values.astype(float)
        if len(users) < SEAM + 2:
            continue
        d = np.abs(np.diff(users))
        seam = d[SEAM - 1]
        others = np.delete(d, SEAM - 1)
        seam_deltas.append(seam)
        other_deltas.append(others.mean())
        seam_pctiles.append(float((others <= seam).mean()))

    seam_deltas = np.array(seam_deltas)
    other_deltas = np.array(other_deltas)
    seam_pctiles = np.array(seam_pctiles)

    n = 48
    starts = np.arange(0, n - SEQ_LEN - PRED_LEN + 1)
    crossing = [(s < SEAM) and (SEAM < s + SEQ_LEN + PRED_LEN) for s in starts]
    train_starts = starts[starts + SEQ_LEN < int(n * TRAIN_RATIO)]
    train_crossing = [(s < SEAM) and (SEAM < s + SEQ_LEN + PRED_LEN) for s in train_starts]

    print(f'stations analyzed: {len(seam_deltas)}')
    print(f'mean |delta| at seam:        {seam_deltas.mean():.2f} users')
    print(f'mean |delta| elsewhere:      {other_deltas.mean():.2f} users')
    print(f'ratio seam/elsewhere:        {seam_deltas.mean() / other_deltas.mean():.2f}x')
    print(f'median percentile of the seam delta within the station\'s own '
          f'deltas: {np.median(seam_pctiles):.0%}')
    print(f'stations where the seam is the single largest jump: '
          f'{(seam_pctiles == 1.0).mean():.1%}')
    print(f'windows crossing the seam: {sum(crossing)}/{len(starts)} of all '
          f'({sum(crossing)/len(starts):.0%}), '
          f'{sum(train_crossing)}/{len(train_starts)} of training '
          f'({sum(train_crossing)/max(len(train_starts),1):.0%})')


if __name__ == '__main__':
    main()
