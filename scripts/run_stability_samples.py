"""Д8 stability check: do the Set 2 conclusions depend on "the first 100"?

Draws N independent stratified random samples of 100 stations (50 4G + 50 5G)
from the full parsed NetData set, EXCLUDING the primary first-100 selection,
and runs the standard base campaign (4 models, no transfer, one seed) on each
sample. Comparing the per-model aggregate metrics across samples against the
primary campaign quantifies how sensitive the Set 2 results are to the
station selection (thesis appendix + one sentence in §6.1).

Sampling is deterministic: numpy default_rng(sample_index * 1000 + seed).

Results: master_thesis_final/stability_sample<i>_s<seed>_<ts>/<model>/...
(standard layout + stations.txt with the drawn station list).

Usage: python scripts/run_stability_samples.py [--samples 3] [--seed 2021]
"""

import argparse
import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from run import SHARED_ARGS, MODEL_CONFIGS, DATA_MODES, build_setting, save_run_config  # noqa: E402
from utils.parser import Parser  # noqa: E402
from exp.exp_main import Exp_Main  # noqa: E402

MODELS = ['TST', 'CATS', 'DeformableTST', 'PerimidFormer']
PER_TECH = 50


def build_args(model, fp, results_dir, use_gpu, seed):
    args_data = copy.deepcopy(SHARED_ARGS)
    args_data.update(copy.deepcopy(MODEL_CONFIGS[model]))
    args_data.update(copy.deepcopy(DATA_MODES['big_data']['overrides']))
    args_data.update({
        'model': model, 'use_gpu': use_gpu, 'random_seed': seed,
        'transfer': False, 'transfer_save': False,
        'data_path': fp['data_path'], 'root_path': fp['root_path'],
        'results_dir': results_dir,
        'checkpoints': os.path.join(results_dir, 'checkpoints'),
    })
    return Parser(args_data, model).args


def draw_sample(all_files, primary, rng):
    """50 + 50 random 4G/5G stations, disjoint from the primary 100."""
    primary_names = {fp['file_name'] for fp in primary}
    pool = {'4G': [], '5G': []}
    for fp in all_files:
        if fp['file_name'] in primary_names:
            continue
        tech = '4G' if '4G' in fp['root_path'] else '5G'
        pool[tech].append(fp)
    sample = []
    for tech in ('4G', '5G'):
        idx = rng.choice(len(pool[tech]), size=PER_TECH, replace=False)
        sample += [pool[tech][i] for i in sorted(idx)]
    return sample


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--samples', type=int, default=3)
    cli.add_argument('--seed', type=int, default=2021)
    cli_args = cli.parse_args()
    sys.argv = sys.argv[:1]      # utils.parser.Parser re-parses argv

    use_gpu = torch.cuda.is_available()
    seed = cli_args.seed
    all_files = DATA_MODES['big_data']['file_lister']()
    primary = all_files[:100]
    print(f'pool: {len(all_files)} stations, primary excluded: {len(primary)}')

    for s in range(1, cli_args.samples + 1):
        rng = np.random.default_rng(s * 1000 + seed)
        sample = draw_sample(all_files, primary, rng)
        run_dir = os.path.join('master_thesis_final',
                               f'stability_sample{s}_s{seed}_' + time.strftime('%Y%m%d_%H%M%S'))
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'command.txt'), 'w') as f:
            f.write(f'python scripts/run_stability_samples.py  # sample {s}\n')
        with open(os.path.join(run_dir, 'stations.txt'), 'w') as f:
            f.write('\n'.join(fp['file_name'] for fp in sample) + '\n')
        print(f'== sample {s}: 100 stations -> {run_dir}')

        for model in MODELS:
            results_dir = os.path.join(run_dir, model)
            for fp in sample:
                args = build_args(model, fp, results_dir, use_gpu, seed)
                setting = build_setting(model, args, fp['file_name'])
                save_run_config(args, setting, results_dir)
                exp = Exp_Main(args)
                exp.train(setting)
                exp.test(setting)
                if use_gpu:
                    torch.cuda.empty_cache()
    print('stability samples done')


if __name__ == '__main__':
    main()
