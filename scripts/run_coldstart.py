"""Cold-start evaluation: leave-one-fold-out over the 91 Set 1 stations.

The 91 stations are split into K=5 volume-stratified folds (per category:
stations sorted by mean training volume, assigned round-robin — fully
deterministic). For every fold k and every model:

  1. donor chain — a sequential transfer chain is trained over the ~73
     stations NOT in fold k (train only, no test); its final weights act as
     the "knowledge of the existing network".
  2. zero-shot — every held-out station of fold k is evaluated on its test
     split with the donor weights as-is (no training on the station at all).
  3. fine-tune — every held-out station warm-starts from the donor weights
     and trains normally on its own training split, then is tested.

From-scratch reference numbers come from the base campaigns (same seed), so
this script does not retrain them. Results land in
master_thesis_final/coldstart_k5_s<seed>_<ts>/fold<k>/<model>_{zeroshot,finetune}/
in the standard layout (configs/results/result.txt), so analysis/ tools work.

Usage: python scripts/run_coldstart.py [--seed 2021] [--folds 5]
"""

import argparse
import copy
import os
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


def build_args(model, fp, results_dir, use_gpu, seed, transfer, transfer_save):
    args_data = copy.deepcopy(SHARED_ARGS)
    args_data.update(copy.deepcopy(MODEL_CONFIGS[model]))
    args_data.update({
        'model': model, 'use_gpu': use_gpu, 'random_seed': seed,
        'transfer': transfer, 'transfer_save': transfer_save,
        'data_path': fp['data_path'], 'root_path': fp['root_path'],
        'results_dir': results_dir,
        'checkpoints': os.path.join(results_dir, 'checkpoints'),
    })
    return Parser(args_data, model).args


def stratified_folds(k):
    """Deterministic volume-stratified folds: per category, sort by mean
    train volume and deal round-robin into k folds."""
    files = DATA_MODES['bs']['file_lister']()
    by_cat = {}
    for fp in files:
        by_cat.setdefault(os.path.basename(os.path.normpath(fp['root_path'])), []).append(fp)
    train_ratio = float(SHARED_ARGS['ratios'].split(',')[0])
    folds = [[] for _ in range(k)]
    for cat, fps in sorted(by_cat.items()):
        vols = [pd.read_csv(os.path.join(fp['root_path'], fp['data_path']))['users']
                .values[:int(192 * train_ratio)].mean() for fp in fps]
        for i, j in enumerate(np.argsort(vols)):
            folds[i % k].append(fps[j])
    return folds


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--seed', type=int, default=2021)
    cli.add_argument('--folds', type=int, default=5)
    cli_args = cli.parse_args()
    sys.argv = sys.argv[:1]      # utils.parser.Parser re-parses argv

    use_gpu = torch.cuda.is_available()
    seed = cli_args.seed
    folds = stratified_folds(cli_args.folds)

    run_dir = os.path.join('master_thesis_final',
                           f'coldstart_k{cli_args.folds}_s{seed}_' + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'command.txt'), 'w') as f:
        f.write('python scripts/run_coldstart.py '
                f'--seed {seed} --folds {cli_args.folds}\n')
    with open(os.path.join(run_dir, 'folds.txt'), 'w') as f:
        for i, fold in enumerate(folds):
            f.write(f'fold{i}: ' + ', '.join(fp['file_name'] for fp in fold) + '\n')
    print(f'{len(folds)} folds, sizes {[len(f) for f in folds]} -> {run_dir}')

    for ki, fold in enumerate(folds):
        donors = [fp for j, f in enumerate(folds) if j != ki for fp in f]
        fold_dir = os.path.join(run_dir, f'fold{ki}')
        for model in MODELS:
            donor_dir = os.path.join(fold_dir, f'{model}_donor')
            print(f'== fold{ki} {model}: donor chain over {len(donors)} stations')
            for fp in donors:
                args = build_args(model, fp, donor_dir, use_gpu, seed,
                                  transfer=True, transfer_save=True)
                exp = Exp_Main(args)
                exp.train(f'donor_{fp["file_name"]}')
                if use_gpu:
                    torch.cuda.empty_cache()
            donor_ckpt = os.path.join(donor_dir, f'transfer_model_{model}.pth')

            print(f'== fold{ki} {model}: zero-shot + fine-tune on {len(fold)} held-out stations')
            for fp in fold:
                # zero-shot: donor weights, no training on the station
                zs_dir = os.path.join(fold_dir, f'{model}_zeroshot')
                args = build_args(model, fp, zs_dir, use_gpu, seed,
                                  transfer=False, transfer_save=False)
                setting = build_setting(model, args, fp['file_name'])
                save_run_config(args, setting, zs_dir)
                exp = Exp_Main(args)
                exp.model.load_state_dict(torch.load(donor_ckpt, map_location=exp.device))
                exp.test(setting)

                # fine-tune: warm-start from donor weights, normal training
                ft_dir = os.path.join(fold_dir, f'{model}_finetune')
                args = build_args(model, fp, ft_dir, use_gpu, seed,
                                  transfer=True, transfer_save=False)
                os.makedirs(ft_dir, exist_ok=True)
                ft_ckpt = os.path.join(ft_dir, f'transfer_model_{model}.pth')
                if not os.path.exists(ft_ckpt):
                    import shutil
                    shutil.copy(donor_ckpt, ft_ckpt)
                setting = build_setting(model, args, fp['file_name'])
                save_run_config(args, setting, ft_dir)
                exp = Exp_Main(args)
                exp.train(setting)
                exp.test(setting)
                if use_gpu:
                    torch.cuda.empty_cache()
    print('cold-start done')


if __name__ == '__main__':
    main()
