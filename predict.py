"""Reuse a trained checkpoint for new predictions.

Every run of run.py stores, per trained model instance:
  <run_dir>/<model>/configs/<setting>.json       full training configuration
  <run_dir>/<model>/checkpoints/<setting>/checkpoint.pth   best weights

This script rebuilds the exact model from the saved config, loads the
checkpoint and forecasts the pred_len hours that follow the end of a CSV
(same format as the training data: time column + features + users).

Examples:
    # forecast the next hour for the station the model was trained on
    python predict.py --config result_data_20260705_120000/TST/configs/<setting>.json

    # forecast for different data (same column layout!)
    python predict.py --config .../configs/<setting>.json --data data/small_bs/bs_2493.csv

    # explicit checkpoint (e.g. after moving folders) + save to file
    python predict.py --config <setting>.json --checkpoint path/to/checkpoint.pth --output forecast.csv
"""

import argparse
import json
import os

import numpy as np
import torch

from exp.exp_main import Exp_Main
from data_provider.data_factory import data_provider


def load_trained_model(config_path, checkpoint=None):
    """Rebuild the model described by a saved config JSON and load its weights.

    Returns (exp, args): an Exp_Main with the trained weights in exp.model.
    """
    with open(config_path) as f:
        config = json.load(f)

    args = argparse.Namespace(**config)
    args.use_gpu = torch.cuda.is_available() and config.get('use_gpu', True)
    args.transfer = False  # never warm-start at inference time

    exp = Exp_Main(args)

    if checkpoint is None:
        checkpoint = os.path.join(args.checkpoints, args.setting, 'checkpoint.pth')
        if not os.path.exists(checkpoint):
            # config was moved together with its run folder: derive the path
            # relative to the config file (configs/ and checkpoints/ are siblings)
            model_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
            checkpoint = os.path.join(model_dir, 'checkpoints', args.setting, 'checkpoint.pth')
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'checkpoint not found: {checkpoint} '
                                f'(pass --checkpoint explicitly)')

    exp.model.load_state_dict(torch.load(checkpoint, map_location=exp.device))
    exp.model.eval()
    print(f'Loaded {args.model} from {checkpoint}')
    return exp, args


def predict_next(exp, args, csv_path=None):
    """Forecast the args.pred_len hours after the last row of csv_path.

    csv_path defaults to the file the model was trained on. The CSV must have
    the training layout: a time column plus the same feature columns
    (the model expects exactly the channel count it was trained with).

    Returns a numpy array of predicted target values, length pred_len.
    """
    if csv_path is not None:
        root_path, data_path = os.path.split(os.path.abspath(csv_path))
        args.root_path, args.data_path = root_path, data_path

    pred_data, pred_loader = data_provider(args, 'pred')

    preds = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in pred_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[2]]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)

            outputs = exp._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -args.pred_len:, -1]  # target channel (users)
            preds.append(outputs.detach().cpu().numpy())

    return np.concatenate(preds, axis=0).reshape(-1)


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--config', required=True,
                     help='path to a saved <run_dir>/<model>/configs/<setting>.json')
    cli.add_argument('--data', default=None,
                     help='CSV to forecast from (default: the file the model was trained on)')
    cli.add_argument('--checkpoint', default=None,
                     help='explicit checkpoint.pth (default: derived from the config)')
    cli.add_argument('--output', default=None,
                     help='optional output file (.csv or .npy) for the forecast')
    cli_args = cli.parse_args()

    exp, args = load_trained_model(cli_args.config, cli_args.checkpoint)
    forecast = predict_next(exp, args, cli_args.data)

    source = cli_args.data or os.path.join(args.root_path, args.data_path)
    print(f'Forecast for the {args.pred_len} hour(s) after the end of {source}:')
    for h, value in enumerate(forecast, start=1):
        print(f'  +{h}h  {args.target} = {value:.2f}')

    if cli_args.output:
        if cli_args.output.endswith('.npy'):
            np.save(cli_args.output, forecast)
        else:
            with open(cli_args.output, 'w') as f:
                f.write(f'horizon_hours,{args.target}\n')
                for h, value in enumerate(forecast, start=1):
                    f.write(f'{h},{value}\n')
        print(f'Saved forecast to {cli_args.output}')


if __name__ == '__main__':
    main()
