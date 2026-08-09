"""Validation-only hyperparameter search (thesis §8.4 reconstruction).

Trains every (config, station, model) combination on the train split and
scores it by the best validation loss — the test split is never touched, so
the search cannot leak into the reported results. The grid varies the three
axes that dominate capacity/regularization/optimization:

    d_model   8 / 16 / 32     (d_ff = 8 * d_model, DeformableTST dims follow)
    dropout   0.2 / 0.4
    lr        1e-4 / 5e-4

= 12 configurations, identical for all four models (the shared-protocol
requirement of the thesis). DeformableTST additionally runs a protocol-parity
arm: every config once with its published recipe (AdamW + warmup/cosine) and
once with the shared protocol (Adam + type3 decay), same epoch budget.

Stations: 3 per category (big/medium/small) at the 25th/50th/75th percentile
of mean train-split `users` volume — deterministic and documented.

Output (under master_thesis_final/hp_search_<timestamp>/):
    hp_results.csv    one row per training: model, arm, config, station, best
                      validation MSE, epochs trained, minutes
    hp_summary.csv    per (model, arm, config): mean/median val MSE + rank
    winner per model and the best shared config by mean rank, printed at the end.

Usage:
    python scripts/hp_search.py [--seed 2021] [--stations 9]
"""

import argparse
import copy
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from run import SHARED_ARGS, MODEL_CONFIGS, DATA_MODES, build_setting  # noqa: E402
from utils.parser import Parser  # noqa: E402
from exp.exp_main import Exp_Main  # noqa: E402

GRID = [
    {'d_model': dm, 'dropout': dr, 'learning_rate': lr}
    for dm, dr, lr in itertools.product([8, 16, 32], [0.2, 0.4], [1e-4, 5e-4])
]

MODELS = ['TST', 'CATS', 'DeformableTST', 'PerimidFormer']


def config_id(cfg):
    return f"dm{cfg['d_model']}_dr{cfg['dropout']}_lr{cfg['learning_rate']:g}"


def pick_stations(per_category=3):
    """Per category, stations at the 25/50/75th percentile of train volume."""
    file_paths = DATA_MODES['bs']['file_lister']()
    by_cat = {}
    for fp in file_paths:
        cat = os.path.basename(os.path.normpath(fp['root_path']))
        by_cat.setdefault(cat, []).append(fp)

    train_ratio = float(SHARED_ARGS['ratios'].split(',')[0])
    chosen = []
    for cat, fps in sorted(by_cat.items()):
        vols = []
        for fp in fps:
            df = pd.read_csv(os.path.join(fp['root_path'], fp['data_path']))
            n = int(len(df) * train_ratio)
            vols.append(df['users'].values[:n].mean())
        order = np.argsort(vols)
        qs = np.linspace(0.25, 0.75, per_category)
        idx = sorted({order[int(round(q * (len(fps) - 1)))] for q in qs})
        for i in idx:
            chosen.append(fps[i])
        print(f'{cat}: ' + ', '.join(fps[i]['file_name'] for i in idx))
    return chosen


def apply_config(args_data, model, cfg):
    """Grid values + the couplings that keep every model consistent."""
    args_data.update(cfg)
    args_data['d_ff'] = 8 * cfg['d_model']
    if model == 'DeformableTST':
        args_data['dims'] = [cfg['d_model']]     # dims == d_model
        args_data['expansion'] = 8               # FFN width ratio, = d_ff/d_model
    return args_data


def build_args(model, fp, results_dir, use_gpu, cfg, arm, seed):
    args_data = copy.deepcopy(SHARED_ARGS)
    args_data.update(copy.deepcopy(MODEL_CONFIGS[model]))
    args_data = apply_config(args_data, model, cfg)
    if arm == 'shared':
        # budget-parity arm: strip the paper recipe, use the shared protocol
        args_data.update({'optimizer': 'Adam', 'lradj': 'type3', 'warmup_epochs': 0})
    args_data.update({
        'model': model,
        'use_gpu': use_gpu,
        'random_seed': seed,
        'transfer': False,
        'transfer_save': False,
        'data_path': fp['data_path'],
        'root_path': fp['root_path'],
        'results_dir': results_dir,
        'checkpoints': os.path.join(results_dir, 'checkpoints'),
    })
    return Parser(args_data, model).args


def best_vali_loss(exp):
    """Validation loss of the best (early-stopping) weights, already loaded."""
    vali_data, vali_loader = exp._get_data(flag='val')
    return float(exp.vali(vali_data, vali_loader, exp._select_criterion()))


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--seed', type=int, default=2021)
    cli.add_argument('--stations', type=int, default=9,
                     help='total stations (3 categories x N/3)')
    cli.add_argument('--out', default=None,
                     help='output folder (default master_thesis_final/hp_search_<ts>)')
    cli_args = cli.parse_args()

    out_root = cli_args.out or os.path.join(
        'master_thesis_final', 'hp_search_' + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, 'command.txt'), 'w') as f:
        f.write('python ' + ' '.join(sys.argv) + '\n')

    use_gpu = torch.cuda.is_available()
    stations = pick_stations(max(1, cli_args.stations // 3))

    jobs = []
    for model in MODELS:
        arms = ['paper', 'shared'] if model == 'DeformableTST' else ['paper']
        for arm, cfg, fp in itertools.product(arms, GRID, stations):
            jobs.append((model, arm, cfg, fp))
    print(f'{len(jobs)} trainings '
          f'({len(GRID)} configs x {len(stations)} stations, '
          f'DeformableTST twice for protocol parity)')

    results_path = os.path.join(out_root, 'hp_results.csv')
    done = set()
    if os.path.exists(results_path):        # resumable after interruption
        prev = pd.read_csv(results_path)
        done = set(zip(prev['model'], prev['arm'], prev['config'], prev['station']))
        print(f'resuming: {len(done)} trainings already recorded')

    for k, (model, arm, cfg, fp) in enumerate(jobs, 1):
        cid = config_id(cfg)
        key = (model, arm, cid, fp['file_name'])
        if key in done:
            continue
        results_dir = os.path.join(out_root, model + ('_shared' if arm == 'shared' else ''))
        args = build_args(model, fp, results_dir, use_gpu, cfg, arm, cli_args.seed)
        setting = cid + '_' + build_setting(model, args, fp['file_name'])

        t0 = time.time()
        exp = Exp_Main(args)
        exp.train(setting)
        loss = best_vali_loss(exp)
        minutes = (time.time() - t0) / 60

        row = pd.DataFrame([{
            'model': model, 'arm': arm, 'config': cid,
            'd_model': cfg['d_model'], 'dropout': cfg['dropout'],
            'lr': cfg['learning_rate'], 'station': fp['file_name'],
            'vali_mse': loss, 'minutes': round(minutes, 2),
        }])
        row.to_csv(results_path, mode='a', index=False,
                   header=not os.path.exists(results_path))
        print(f'[{k}/{len(jobs)}] {model}/{arm} {cid} {fp["file_name"]}: '
              f'vali_mse={loss:.4f} ({minutes:.1f} min)')
        if use_gpu:
            torch.cuda.empty_cache()

    # ------------------------------------------------------------- summary
    df = pd.read_csv(results_path)
    # per station, configs are comparable; normalize by the station's best
    # so easy/hard stations weigh equally, then aggregate
    df['rel'] = df.groupby(['model', 'arm', 'station'])['vali_mse'] \
                  .transform(lambda s: s / s.min())
    summary = (df.groupby(['model', 'arm', 'config'])
                 .agg(mean_vali=('vali_mse', 'mean'),
                      median_vali=('vali_mse', 'median'),
                      mean_rel=('rel', 'mean'))
                 .reset_index())
    summary['rank_in_model'] = summary.groupby(['model', 'arm'])['mean_rel'] \
                                      .rank(method='min')
    summary.to_csv(os.path.join(out_root, 'hp_summary.csv'), index=False)

    print('\n=== winners per model (by mean relative vali MSE) ===')
    for (model, arm), g in summary.groupby(['model', 'arm']):
        best = g.loc[g['mean_rel'].idxmin()]
        print(f'{model:>14}/{arm}: {best["config"]}  '
              f'(mean_rel {best["mean_rel"]:.3f}, mean vali {best["mean_vali"]:.2f})')

    shared = (summary[summary['arm'] == 'paper']
              .groupby('config')['mean_rel'].mean().sort_values())
    print('\n=== best SHARED config across the four models ===')
    for cid, score in shared.head(3).items():
        print(f'  {cid}: mean relative vali MSE {score:.3f}')
    print(f'\nsummary: {os.path.join(out_root, "hp_summary.csv")}')


if __name__ == '__main__':
    main()
