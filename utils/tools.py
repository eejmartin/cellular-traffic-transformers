import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from torch import optim
import os
import glob
import re
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


"""class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean"""


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def read_file_names_by_order():
    base_dir = os.path.join(os.getcwd(), 'data')
    folders = {
        'big': os.path.join(base_dir, 'big_bs'),
        'small': os.path.join(base_dir, 'small_bs'),
        'medium': os.path.join(base_dir, 'medium_bs')
    }

    # Collect and sort files
    file_objects = {'big': [], 'small': [], 'medium': []}

    for size in folders:
        folder_path = folders[size]
        folder_files = [f for f in glob.glob(os.path.join(folder_path, '*')) 
             if os.path.isfile(f)]
        
         # Sort files using natural order
        file_objects[size] = sorted(folder_files, key=lambda x: natural_sort_key(os.path.basename(x)))        
    
    # Process in round-robin order
    max_files = max(len(file_objects['big']),  
                    len(file_objects['small']),
                    len(file_objects['medium']))

    file_paths = []

    for i in range(max_files):
        for size in ['big', 'medium', 'small']:
            if i < len(file_objects[size]):
                entry = file_objects[size][i]
                file_paths.append({
                    'root_path': os.path.dirname(entry),
                    'data_path': os.path.basename(entry),
                    'file_name': os.path.basename(entry).split('.')[0],
                    })


    return file_paths

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def read_big_data_file_names():
    """Per-station files of the concatenated big_data set (weekday+weekend,
    48 hourly rows), produced by parse_big_data.py. Round-robin 4G/5G.

    file_name is prefixed with the technology (e.g. 4G_bs_10055) so output
    folders never collide between the two sets.
    """
    base_dir = os.path.join(os.getcwd(), 'data', 'big_data', 'parsed', 'concat')
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f'{base_dir} not found — run "python parse_big_data.py" first (see README)')

    file_objects = {}
    for tech in ['4G', '5G']:
        folder = os.path.join(base_dir, tech)
        files = [f for f in glob.glob(os.path.join(folder, 'bs_*.csv')) if os.path.isfile(f)]
        file_objects[tech] = sorted(files, key=lambda x: natural_sort_key(os.path.basename(x)))

    file_paths = []
    max_files = max(len(v) for v in file_objects.values())
    for i in range(max_files):
        for tech in ['4G', '5G']:
            if i < len(file_objects[tech]):
                entry = file_objects[tech][i]
                file_paths.append({
                    'root_path': os.path.dirname(entry),
                    'data_path': os.path.basename(entry),
                    'file_name': tech + '_' + os.path.basename(entry).split('.')[0],
                })

    return file_paths

def plot_result_files(base_dir=None):
    if base_dir is None:
        base_dir = os.path.join(os.getcwd(), 'result_data/results')

    full_paths = []

    # Get all subdirectories in the base directory
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    folders.sort()  # Sort folders alphabetically

    for folder in folders:
        full_paths.append({
            "file_path": base_dir + '/' + folder,
            'file': folder
        })

    return full_paths

def parse_ratios(ratio_string):
    try:
        ratios = [float(x) for x in ratio_string.split(',')]
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("Three values are required and they must sum to 1.")
        return ratios
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    
def warm_up(optimizer, now_epoch, args):

    lr = args.learning_rate * (now_epoch+1) / args.warmup_epochs
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
        print(f'Updating learning rate to {lr:.7f}')

    return

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x