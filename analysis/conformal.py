"""Split-conformal prediction intervals for finished campaigns (§9.1.8).

For every trained station of a run folder: rebuild the model from its config,
load the checkpoint, collect absolute residuals on the VALIDATION split and
take the finite-sample 90 % quantile as the interval half-width; then measure
on the TEST split how often the true value falls inside prediction ± width
(empirical coverage) and how wide the interval is relative to the station's
mean load.

Outputs <run_dir>_conformal.csv (per station) and a printed per-model
summary: target coverage 0.9 vs empirical, mean width, relative width.

Usage:
    python analysis/conformal.py <run_dir> [<run_dir> ...] [--alpha 0.1] [--cpu]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from exp.exp_main import Exp_Main  # noqa: E402
from data_provider.data_factory import data_provider  # noqa: E402


def collect_preds(exp, args, flag):
    _, loader = data_provider(args, flag)
    preds, trues = [], []
    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp],
                                dim=1).float().to(exp.device)
            out = exp._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            preds.append(out[:, -args.pred_len:, f_dim:].detach().cpu().numpy())
            trues.append(batch_y[:, -args.pred_len:, f_dim:].detach().cpu().numpy())
    return np.concatenate(preds).reshape(-1), np.concatenate(trues).reshape(-1)


def process_run(run_dir, alpha, use_gpu):
    rows = []
    for model_dir in sorted(glob.glob(os.path.join(run_dir, '*', 'configs'))):
        model_dir = os.path.dirname(model_dir)
        model = os.path.basename(model_dir)
        for cfg_path in sorted(glob.glob(os.path.join(model_dir, 'configs', '*.json'))):
            with open(cfg_path) as f:
                cfg = json.load(f)
            ckpt = os.path.join(model_dir, 'checkpoints', cfg['setting'], 'checkpoint.pth')
            if not os.path.exists(ckpt):
                continue
            args = argparse.Namespace(**cfg)
            args.use_gpu = use_gpu
            args.transfer = False
            try:
                exp = Exp_Main(args)
                exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
                vp, vt = collect_preds(exp, args, 'val')
                tp, tt = collect_preds(exp, args, 'test')
            except Exception as e:
                print(f'  FAILED {cfg["setting"][:60]}: {e}')
                continue
            n = len(vp)
            if n == 0 or len(tp) == 0:
                continue
            # finite-sample split-conformal quantile
            q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
            width = float(np.quantile(np.abs(vp - vt), q_level))
            covered = float(np.mean(np.abs(tp - tt) <= width))
            rows.append({'model': model, 'station': os.path.basename(cfg['data_path']),
                         'half_width': width, 'coverage_test': covered,
                         'n_val': n, 'n_test': len(tp),
                         'rel_width': width / max(float(np.mean(np.abs(tt))), 1e-8)})
            if use_gpu:
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('run_dirs', nargs='+')
    cli.add_argument('--alpha', type=float, default=0.1, help='1 - target coverage')
    cli.add_argument('--cpu', action='store_true')
    args = cli.parse_args()
    use_gpu = torch.cuda.is_available() and not args.cpu

    for rd in args.run_dirs:
        rd = rd.rstrip('/')
        print(f'== {os.path.basename(rd)}')
        df = process_run(rd, args.alpha, use_gpu)
        if df.empty:
            print('  nothing parsed')
            continue
        out = rd + '_conformal.csv'
        df.to_csv(out, index=False)
        summary = df.groupby('model').agg(
            stations=('station', 'size'),
            coverage=('coverage_test', 'mean'),
            mean_half_width=('half_width', 'mean'),
            median_rel_width=('rel_width', 'median')).round(3)
        print(summary.to_string())
        print('  ->', out)


if __name__ == '__main__':
    main()
