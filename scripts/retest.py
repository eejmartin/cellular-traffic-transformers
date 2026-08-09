"""Re-evaluate finished campaigns with the fixed evaluation pipeline.

Rebuilds every trained model instance of a run folder from its saved config
JSON, loads the existing checkpoint (training is NOT repeated) and runs
Exp_Main.test() again — with the corrected test loader (drop_last=False),
the corrected checkpoint path handling and the extended metric set. Use it to
re-score campaigns that were trained before the evaluation fixes, or to
sanity-check that the fixed pipeline reproduces a fresh campaign's numbers.

Output goes into a sibling folder <run_dir>_retest/<model>/ by default
(fresh result.txt, results/<setting>/{metrics.npy,metrics_ext.json,pred.npy,
true.npy}, test_results/), leaving the original run untouched; pass
--in-place to overwrite inside the original run folder instead.

Examples:
    python scripts/retest.py master_thesis_final/old_runs/result_data_20260705_131934
    python scripts/retest.py <run1> <run2> ...          # several runs in one go
    python scripts/retest.py <run> --models TST CATS    # subset of models
"""

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_main import Exp_Main  # noqa: E402


def find_checkpoint(config, config_path):
    """checkpoint.pth for a config, tolerant to the run folder having moved."""
    setting = config['setting']
    candidates = [os.path.join(config['checkpoints'], setting, 'checkpoint.pth')]
    # configs/ and checkpoints/ are siblings inside <run_dir>/<model>/
    model_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    candidates.append(os.path.join(model_dir, 'checkpoints', setting, 'checkpoint.pth'))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def retest_model_dir(model_dir, out_dir, use_gpu):
    """Re-run test() for every configs/*.json of one <run_dir>/<model>/."""
    config_paths = sorted(glob.glob(os.path.join(model_dir, 'configs', '*.json')))
    done = failed = 0
    for config_path in config_paths:
        with open(config_path) as f:
            config = json.load(f)

        checkpoint = find_checkpoint(config, config_path)
        if checkpoint is None:
            print(f'  MISSING checkpoint for {config["setting"]}')
            failed += 1
            continue

        args = argparse.Namespace(**config)
        args.use_gpu = use_gpu and config.get('use_gpu', True)
        args.transfer = False          # never warm-start at evaluation time
        args.results_dir = out_dir     # fresh result.txt / results/ tree
        args.test_flop = False

        try:
            exp = Exp_Main(args)
            exp.model.load_state_dict(torch.load(checkpoint, map_location=exp.device))
            exp.test(config['setting'])
            done += 1
        except Exception as e:
            print(f'  FAILED {config["setting"]}: {e}')
            failed += 1
        finally:
            if use_gpu:
                torch.cuda.empty_cache()
    return done, failed


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('run_dirs', nargs='+',
                     help='campaign folder(s): <run_dir>/<model>/configs/*.json')
    cli.add_argument('--models', nargs='+', default=None,
                     help='only these model subfolders (default: all found)')
    cli.add_argument('--in-place', action='store_true',
                     help='write into the original run folder instead of <run_dir>_retest '
                          '(overwrites result.txt and results/)')
    cli.add_argument('--cpu', action='store_true', help='force CPU evaluation')
    cli_args = cli.parse_args()

    use_gpu = torch.cuda.is_available() and not cli_args.cpu

    for run_dir in cli_args.run_dirs:
        run_dir = run_dir.rstrip('/')
        if not os.path.isdir(run_dir):
            print(f'skip {run_dir}: not a directory')
            continue

        model_dirs = [d for d in sorted(os.listdir(run_dir))
                      if os.path.isdir(os.path.join(run_dir, d, 'configs'))]
        if cli_args.models:
            model_dirs = [d for d in model_dirs if d in cli_args.models]

        out_root = run_dir if cli_args.in_place else run_dir + '_retest'
        print(f'== {run_dir} -> {out_root}  (models: {", ".join(model_dirs) or "none"})')

        total_done = total_failed = 0
        for model in model_dirs:
            out_dir = os.path.join(out_root, model)
            if not cli_args.in_place:
                # start from a clean result.txt: test() appends
                os.makedirs(out_dir, exist_ok=True)
                result_txt = os.path.join(out_dir, 'result.txt')
                if os.path.exists(result_txt):
                    os.remove(result_txt)
            done, failed = retest_model_dir(os.path.join(run_dir, model), out_dir, use_gpu)
            print(f'  {model}: {done} re-evaluated, {failed} failed')
            total_done += done
            total_failed += failed
        print(f'== {run_dir}: {total_done} OK, {total_failed} failed')


if __name__ == '__main__':
    main()
