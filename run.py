import os
import sys
import copy
import json
import time
import argparse
import torch
import numpy as np
from exp.exp_main import Exp_Main
from utils.parser import Parser
from utils.tools import read_file_names_by_order, read_big_data_file_names, plot_result_files
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Shared configuration
#
# Every parameter in SHARED_ARGS is identical for all four models so results
# are directly comparable between architectures. Per-model entries in
# MODEL_CONFIGS contain only what is architecture-specific (see models/README.md
# for an explanation of every parameter).
# ---------------------------------------------------------------------------

SHARED_ARGS = {
    # data
    "data": "custom",
    "features": "MS",          # multivariate input (packets, bytes, users) -> univariate output (users)
    "target": "users",
    "freq": "h",
    "ratios": "0.7,0.1,0.2",   # train / validation / test split
    "scale": False,            # False = keep raw units (all models normalize per-window internally)
    # forecasting task
    "seq_len": 24,             # one day of hourly history
    "label_len": 1,
    "pred_len": 1,             # predict the next hour
    # channel counts (all must equal the number of input features)
    "enc_in": 3,               # used by PatchTST
    "dec_in": 3,               # used by CATS
    "n_vars": 3,               # used by DeformableTST
    "chan_in": 3,              # used by PerimidFormer
    # shared architecture scale — selected by the validation-only grid search
    # (scripts/hp_search.py, 12 configs x 9 stations x 4 models, seed 2021):
    # dm32_dr0.2_lr0.0005 ranked best shared config (mean relative vali MSE
    # 1.119; the previous dm16_dr0.4_lr0.0001 ranked 10th of 12)
    "d_model": 32,
    "n_heads": 4,
    "e_layers": 3,             # PatchTST encoder depth
    "d_layers": 3,             # CATS decoder depth (its only stack)
    "d_ff": 256,
    "dropout": 0.2,
    "fc_dropout": 0.05,
    "head_dropout": 0.0,
    # patching (PatchTST & CATS)
    "patch_len": 6,
    "stride": 1,
    "padding_patch": "end",
    # normalization heads
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "individual": 0,
    # training protocol (identical for all models)
    "train_epochs": 100,
    "batch_size": 6,
    "patience": 50,
    "learning_rate": 0.0005,   # from the grid search (see above)
    "lradj": "type3",
    "pct_start": 0.3,
    "optimizer": "Adam",
    "warmup_epochs": 0,
    "weight_decay": 0.05,      # only used when optimizer == 'AdamW'
    "num_workers": 0,
    # misc
    "embed": "timeF",
    "factor": 1,
    "distil": True,
    "des": "Exp",
    "output_attention": False,
    "test_flop": False,
    "transfer": False,         # True = warm-start each base station from the previous
                               # one; also switchable per run with the -t CLI flag
    "task_name": "long_term_forecast",
    "gpu": 0,
    "use_multi_gpu": False,
}

# Architecture-specific settings. Training-protocol overrides marked with
# (paper recipe) reproduce the original repository's published training setup;
# remove them to train every model with the fully identical shared protocol.
MODEL_CONFIGS = {
    "TST": {
        "model_id": "PatchTST",
    },
    "CATS": {
        "model_id": "CATS",
        # QAM_start / QAM_end / query_independence come from parser defaults
        # (0.1 / 0.3 / True), matching the original repository.
    },
    "DeformableTST": {
        "model_id": "DeformableTST",
        # Single-stage configuration: the original authors use one stage with
        # stem_ratio=1 for short inputs (seq_len ~ 24); their 4-stage default
        # would downsample 24 steps to 3. Depth 3 mirrors e_layers=3.
        "stem_ratio": 1,
        "down_ratio": 2,
        "fmap_size": 24,               # sizes the relative-position-bias table
        "dims": [32],                  # = d_model
        "depths": [3],
        "stage_spec": [["D", "D", "D"]],
        "heads": [4],                  # = n_heads
        "expansion": 8,                # FFN width: 32 * 8 = 256 = d_ff
        "drop_path_rate": 0.3,
        "layer_scale_value": [-1],
        "use_pe": [1],
        "use_lpu": [1],
        "local_kernel_size": [3],
        "use_dwc_mlp": [1],
        "window_size": [3],
        "nat_ksize": [3],
        "ksize": [3],                  # offset sub-network kernel
        "stride": [1],                 # offset sub-network stride (per stage)
        "n_groups": [2],
        "offset_range_factor": [-1],
        "no_off": [0],
        "dwc_pe": [0],
        "fixed_pe": [0],
        "log_cpb": [0],
        "revin_affine": 0,
        "revin_subtract_last": 0,
        "head_type": "Flatten",
        "use_head_norm": 1,
        # (paper recipe) AdamW + weight decay, linear warmup + cosine annealing
        "optimizer": "AdamW",
        "lradj": "cos",
        "warmup_epochs": 5,
    },
    "PerimidFormer": {
        "model_id": "PerimidFormer",
        "layers": 3,                   # encoder depth, mirrors e_layers=3
        "top_k": 2,                    # number of dominant FFT periods
        "moving_avg": 25,              # trend/seasonal decomposition kernel
    },
}


# Dataset modes, selected on the command line: `python run.py -f big_data`.
# Every mode trains one model instance per base station; only the data source
# and the window length differ. The big_data series are 48 hourly rows
# (weekday+weekend day, see parse_big_data.py), so seq_len shrinks to 12 —
# with the 0.7/0.1/0.2 split that yields 21 train / 6 val / 9 test windows.
DATA_MODES = {
    'bs': {
        'file_lister': read_file_names_by_order,
        'overrides': {},
    },
    'big_data': {
        'file_lister': read_big_data_file_names,
        'overrides': {
            'seq_len': 12,
            'fmap_size': 12,   # DeformableTST position-bias table ~ seq_len
        },
    },
}


def build_setting(model, args, file_name):
    if model == 'TST':
        return '{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
            args.factor, args.embed, args.distil, args.des, file_name)
    if model == 'CATS':
        return '{}_{}_{}_sl{}_pl{}_dm{}_nh{}_dl{}_df{}_qi{}_{}'.format(
            args.model_id, args.model, args.data, args.seq_len, args.pred_len,
            args.d_model, args.n_heads, args.d_layers, args.d_ff,
            args.query_independence, file_name)
    if model == 'DeformableTST':
        return '{}_{}_{}_Input_{}_Output_{}_Stem_{}_Dims_{}_FFN_{}_Layer{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.seq_len, args.pred_len,
            args.stem_ratio, args.dims, args.expansion, args.depths,
            args.stage_spec, file_name)
    if model == 'PerimidFormer':
        return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_l{}_k{}_eb{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
            args.layers, args.top_k, args.embed, args.des, file_name)
    raise ValueError(f'No setting format for model {model!r}')


def save_run_config(args, setting, results_dir):
    """Persist the full merged configuration so predict.py can rebuild the
    exact model later (see models/README.md, 'Using trained models later')."""
    config_dir = os.path.join(results_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    config = dict(vars(args))
    config['setting'] = setting
    with open(os.path.join(config_dir, setting + '.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)


def plot_result(run_dir, models):
    for model in models:
        results_base = os.path.join(run_dir, model, 'results')
        if not os.path.isdir(results_base):
            continue

        for folder in plot_result_files(results_base):
            pred_path = os.path.join(folder['file_path'], 'pred.npy')
            true_path = os.path.join(folder['file_path'], 'true.npy')
            if not (os.path.exists(pred_path) and os.path.exists(true_path)):
                print(f"Skipping {folder['file']} (no pred/true arrays)")
                continue

            preds = np.load(pred_path).squeeze()
            trues = np.load(true_path).squeeze()

            plt.figure(figsize=(15, 5))
            plt.plot(trues, label="Реални вредности", color="blue")
            plt.plot(preds, label="Предвидени вредности", color="red")
            plt.title("Реални наспроти предвидени вредности", fontsize=14)
            plt.xlabel("Тест прозорец (час)", fontsize=12)
            plt.ylabel("Број на корисници", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid()

            folder_path = os.path.join(run_dir, model, 'plots')
            os.makedirs(folder_path, exist_ok=True)
            plt.savefig(os.path.join(folder_path, 'true_vs_pred_plot_' + folder['file'] + '.png'))
            plt.close()


def build_args(model, file_path, results_dir, use_gpu, data_mode, transfer,
               transfer_save=True, seed=2021):
    """Fresh merged config per (model, station): nothing leaks between runs."""
    args_data = copy.deepcopy(SHARED_ARGS)
    args_data.update(copy.deepcopy(MODEL_CONFIGS[model]))
    args_data.update(copy.deepcopy(data_mode['overrides']))
    if transfer:
        args_data["transfer"] = True
    args_data["transfer_save"] = transfer_save
    args_data["model"] = model
    args_data["use_gpu"] = use_gpu
    args_data["random_seed"] = seed
    args_data["data_path"] = file_path['data_path']
    args_data["root_path"] = file_path['root_path']
    args_data["results_dir"] = results_dir
    args_data["checkpoints"] = os.path.join(results_dir, 'checkpoints')
    return Parser(args_data, model).args


if __name__ == '__main__':

    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument('-f', '--files', choices=list(DATA_MODES), default='bs')
    cli.add_argument('-l', '--limit', type=int, default=None)
    cli.add_argument('-t', '--transfer', action='store_true',
                     help='enable transfer learning: warm-start each station '
                          'from the previously trained one (per model)')
    cli.add_argument('--transfer-mode', dest='transfer_mode',
                     choices=['sequential', 'ordered', 'grouped', 'federated', 'combined'],
                     default='sequential',
                     help='transfer strategy (implies -t): sequential = file order; '
                          'ordered = similarity-sorted chain; grouped = one chain '
                          'per KMeans cluster of daily profiles; federated = '
                          'FedAvg pretraining, then per-station fine-tuning from '
                          'the shared global weights; combined = all three: '
                          'FedAvg pretraining, then one similarity-ordered chain '
                          'per cluster, each chain seeded from the global weights')
    cli.add_argument('--groups', type=int, default=3,
                     help='number of clusters for --transfer-mode grouped')
    cli.add_argument('--fed-rounds', dest='fed_rounds', type=int, default=5,
                     help='FedAvg rounds for --transfer-mode federated')
    cli.add_argument('--fed-epochs', dest='fed_epochs', type=int, default=5,
                     help='local epochs per station per FedAvg round')
    cli.add_argument('-s', '--seed', type=int, default=2021,
                     help='random seed for weight init / dropout / shuffling; '
                          'encoded in the run folder name for multi-seed campaigns')
    cli_args, _ = cli.parse_known_args()
    data_mode = DATA_MODES[cli_args.files]
    if cli_args.transfer_mode != 'sequential':
        cli_args.transfer = True    # the mode only makes sense with transfer on

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.empty_cache()

    file_paths = data_mode['file_lister']()
    if cli_args.limit is not None:
        file_paths = file_paths[:cli_args.limit]
    print(f"Dataset mode '{cli_args.files}': {len(file_paths)} station files")
    if cli_args.transfer:
        print(f"Transfer learning: ON, mode '{cli_args.transfer_mode}'")

    # station ordering / grouping is data-driven and identical for all models
    station_groups = [file_paths]
    if cli_args.transfer and cli_args.transfer_mode == 'ordered':
        from utils.transfer_strategies import order_by_similarity
        station_groups = [order_by_similarity(file_paths)]
        print('ordered: greedy similarity chain over daily profiles')
    elif cli_args.transfer and cli_args.transfer_mode in ('grouped', 'combined'):
        from utils.transfer_strategies import group_stations
        station_groups = group_stations(file_paths, cli_args.groups)
        print(f'{cli_args.transfer_mode}: {len(station_groups)} clusters, sizes '
              f'{[len(g) for g in station_groups]} (similarity-ordered within)')

    models = ['TST', 'CATS', 'DeformableTST', 'PerimidFormer']

    # every invocation of run.py writes into its own folder, with one
    # subfolder per model; the folder name encodes the command that produced
    # it (dataset, limit, transfer strategy) plus a timestamp for uniqueness
    tag_parts = [cli_args.files]
    if cli_args.limit is not None:
        tag_parts.append(f'l{cli_args.limit}')
    if cli_args.transfer:
        transfer_tag = 'transfer-' + cli_args.transfer_mode
        if cli_args.transfer_mode in ('grouped', 'combined'):
            transfer_tag += f'-g{cli_args.groups}'
        if cli_args.transfer_mode in ('federated', 'combined'):
            transfer_tag += f'-r{cli_args.fed_rounds}-e{cli_args.fed_epochs}'
        tag_parts.append(transfer_tag)
    tag_parts.append(f's{cli_args.seed}')
    # all run folders live under RESULTS_ROOT (git-ignored), keeping the new
    # v2 campaigns clearly separated from the pre-fix results
    RESULTS_ROOT = 'master_thesis_final'
    run_dir = os.path.join(
        RESULTS_ROOT,
        'result_data_' + '_'.join(tag_parts) + '_' + time.strftime('%Y%m%d_%H%M%S'))
    print(f'Results of this run go to: {run_dir}/<model>/')

    # exact command for reproducibility
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'command.txt'), 'w') as f:
        f.write('python ' + ' '.join(sys.argv) + '\n')

    for model in models:
        argsSettings = []
        results_dir = os.path.join(run_dir, model)
        # federated mode: every station fine-tunes from the same global
        # weights, so the per-station trainings must not update the file.
        # combined mode chains within each cluster, so it does update it
        # (and re-seeds from the global weights at every cluster boundary).
        transfer_save = cli_args.transfer_mode != 'federated'

        global_state = None
        if (cli_args.transfer and file_paths
                and cli_args.transfer_mode in ('federated', 'combined')):
            from utils.transfer_strategies import federated_pretrain
            print(f'{cli_args.transfer_mode}: FedAvg pretraining {model} '
                  f'({cli_args.fed_rounds} rounds x {cli_args.fed_epochs} epochs)')
            fed_args = [build_args(model, fp, results_dir, use_gpu, data_mode,
                                   transfer=False, seed=cli_args.seed)
                        for fp in file_paths]
            global_state = federated_pretrain(fed_args,
                                              rounds=cli_args.fed_rounds,
                                              local_epochs=cli_args.fed_epochs)
            os.makedirs(results_dir, exist_ok=True)
            torch.save(global_state,
                       os.path.join(results_dir, f'transfer_model_{model}.pth'))
            torch.cuda.empty_cache()

        for group in station_groups:
            checkpoint = os.path.join(results_dir, f'transfer_model_{model}.pth')
            if cli_args.transfer_mode == 'grouped':
                # each cluster is an independent chain: reset the checkpoint
                if os.path.exists(checkpoint):
                    os.remove(checkpoint)
            elif cli_args.transfer_mode == 'combined' and global_state is not None:
                # each cluster chain starts from the federated global weights
                torch.save(global_state, checkpoint)
                print(f'combined: {model} chain of {len(group)} stations '
                      're-seeded from federated global weights')

            for file_path in group:
                args = build_args(model, file_path, results_dir, use_gpu,
                                  data_mode, cli_args.transfer, transfer_save,
                                  seed=cli_args.seed)

                setting = build_setting(model, args, file_path['file_name'])
                save_run_config(args, setting, results_dir)

                argsSettings.append({
                    'args': args,
                    'setting': setting
                })

                exp = Exp_Main(args)
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

                torch.cuda.empty_cache()

        for value in argsSettings:
            exp = Exp_Main(value['args'])
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(value['setting']))
            exp.predict(value['setting'], True)

            torch.cuda.empty_cache()

    plot_result(run_dir, models)
