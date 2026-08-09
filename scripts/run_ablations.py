"""Ablation campaigns (thesis 10.14 / 10.9): all four models, seeded, using
the exact run.py protocol (train -> test, config JSON + result.txt per run).

Ablations:
    seqlen    seq_len 12 (vs the base campaigns' 24) on Set 1, all 91 stations
    channels  users-only (features=S, 1 channel) vs the base 3 channels,
              Set 1 (91) + Set 2 (100)
    horizon   pred_len in {3, 6, 12} on Set 1, all 91 stations
    history   history length in {48, 96, 144, 192} hours (series cropped to
              the most recent L rows) on 30 volume-stratified Set 1 stations

Each ablation writes master_thesis_final/ablation_<name>[...]_s<seed>_<ts>/
with the standard <model>/{checkpoints,configs,results,...} layout, so
analysis/ and scripts/retest.py work on it unchanged.

Usage:
    python scripts/run_ablations.py all            # everything, in sequence
    python scripts/run_ablations.py seqlen channels horizon history --seed 2021
"""

import argparse
import copy
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from run import SHARED_ARGS, MODEL_CONFIGS, DATA_MODES, build_setting, save_run_config  # noqa: E402
from utils.parser import Parser  # noqa: E402
from exp.exp_main import Exp_Main  # noqa: E402

MODELS = ['TST', 'CATS', 'DeformableTST', 'PerimidFormer']
ROOT = 'master_thesis_final'


def build_args(model, fp, results_dir, use_gpu, seed, overrides):
    args_data = copy.deepcopy(SHARED_ARGS)
    args_data.update(copy.deepcopy(MODEL_CONFIGS[model]))
    args_data.update(copy.deepcopy(overrides))
    args_data.update({
        'model': model, 'use_gpu': use_gpu, 'random_seed': seed,
        'transfer': False, 'transfer_save': False,
        'data_path': fp['data_path'], 'root_path': fp['root_path'],
        'results_dir': results_dir,
        'checkpoints': os.path.join(results_dir, 'checkpoints'),
    })
    return Parser(args_data, model).args


def run_campaign(tag, file_paths, overrides, seed, use_gpu):
    run_dir = os.path.join(ROOT, f'ablation_{tag}_s{seed}_' + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'command.txt'), 'w') as f:
        f.write('python ' + ' '.join(sys.argv) + f'   # campaign {tag}\n')
    print(f'== {tag}: {len(file_paths)} stations x {len(MODELS)} models -> {run_dir}')

    for model in MODELS:
        results_dir = os.path.join(run_dir, model)
        for fp in file_paths:
            args = build_args(model, fp, results_dir, use_gpu, seed, overrides)
            setting = build_setting(model, args, fp['file_name'])
            save_run_config(args, setting, results_dir)
            exp = Exp_Main(args)
            exp.train(setting)
            exp.test(setting)
            if use_gpu:
                torch.cuda.empty_cache()
    return run_dir


def stratified_stations(per_category=10):
    """N per category at evenly spaced volume quantiles (deterministic)."""
    files = DATA_MODES['bs']['file_lister']()
    by_cat = {}
    for fp in files:
        by_cat.setdefault(os.path.basename(os.path.normpath(fp['root_path'])), []).append(fp)
    train_ratio = float(SHARED_ARGS['ratios'].split(',')[0])
    chosen = []
    for cat, fps in sorted(by_cat.items()):
        vols = [pd.read_csv(os.path.join(fp['root_path'], fp['data_path']))['users']
                .values[:int(192 * train_ratio)].mean() for fp in fps]
        order = np.argsort(vols)
        qs = np.linspace(0.05, 0.95, per_category)
        idx = sorted({order[int(round(q * (len(fps) - 1)))] for q in qs})
        chosen += [fps[i] for i in idx]
    return chosen


def crop_dataset(file_paths, hours, dest_root):
    """Copies of the station CSVs truncated to the most recent `hours` rows."""
    cropped = []
    for fp in file_paths:
        cat = os.path.basename(os.path.normpath(fp['root_path']))
        dest_dir = os.path.join(dest_root, f'hist{hours}', cat)
        os.makedirs(dest_dir, exist_ok=True)
        df = pd.read_csv(os.path.join(fp['root_path'], fp['data_path']))
        df.tail(hours).to_csv(os.path.join(dest_dir, fp['data_path']), index=False)
        cropped.append({'root_path': dest_dir, 'data_path': fp['data_path'],
                        'file_name': fp['file_name']})
    return cropped


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('ablations', nargs='+',
                     choices=['all', 'seqlen', 'channels', 'horizon', 'history'])
    cli.add_argument('--seed', type=int, default=2021)
    cli_args = cli.parse_args()
    # utils.parser.Parser re-parses sys.argv on every build_args call and
    # would reject this script's own positional arguments — neutralize it
    sys.argv = sys.argv[:1]
    todo = ['seqlen', 'channels', 'horizon', 'history'] \
        if 'all' in cli_args.ablations else cli_args.ablations

    use_gpu = torch.cuda.is_available()
    bs_files = DATA_MODES['bs']['file_lister']()
    big_files = DATA_MODES['big_data']['file_lister']()[:100]
    seed = cli_args.seed

    if 'seqlen' in todo:
        run_campaign('seqlen12-bs', bs_files,
                     {'seq_len': 12, 'fmap_size': 12}, seed, use_gpu)

    if 'channels' in todo:
        single = {'features': 'S', 'enc_in': 1, 'dec_in': 1, 'n_vars': 1, 'chan_in': 1}
        run_campaign('users-only-bs', bs_files, dict(single), seed, use_gpu)
        run_campaign('users-only-big', big_files,
                     dict(single, seq_len=12, fmap_size=12), seed, use_gpu)

    if 'horizon' in todo:
        for h in (3, 6, 12):
            run_campaign(f'horizon{h}-bs', bs_files, {'pred_len': h}, seed, use_gpu)

    if 'history' in todo:
        stations = stratified_stations(10)
        print('history stations:', [fp['file_name'] for fp in stations])
        crop_root = os.path.join(ROOT, 'ablation_data')
        for hours in (48, 96, 144, 192):
            files = crop_dataset(stations, hours, crop_root)
            overrides = {'seq_len': 12, 'fmap_size': 12} if hours == 48 else {}
            run_campaign(f'history{hours}', files, overrides, seed, use_gpu)
        shutil.rmtree(crop_root, ignore_errors=True)

    print('ablations done:', ', '.join(todo))


if __name__ == '__main__':
    main()
