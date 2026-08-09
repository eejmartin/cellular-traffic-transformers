from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchTST, CATS, DeformableTST, PerimidFormer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, warm_up
from utils.metrics import metric, metric_all

import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def seed_everything(seed):
    """Make weight init, dropout and DataLoader shuffling reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Exp_Main(Exp_Basic):
    """Unified experiment class for all four models.

    Every model is trained/validated/tested through the same loop so results
    are directly comparable. Model-specific behaviour is limited to:
      - how the forward pass is invoked (_model_forward)
      - the optimizer (Adam vs AdamW, selected via args.optimizer)
      - the learning-rate schedule (selected via args.lradj / args.warmup_epochs)
    """

    MODEL_CLASSES = {
        'TST': PatchTST,
        'CATS': CATS,
        'DeformableTST': DeformableTST,
        'PerimidFormer': PerimidFormer,
    }

    def __init__(self, args):
        # seed before the model is built (Exp_Basic.__init__ builds it), so
        # weight init, dropout and shuffling are reproducible per station
        # regardless of how many stations ran before this one
        seed_everything(getattr(args, 'random_seed', 2021))
        super(Exp_Main, self).__init__(args)

    @property
    def _results_dir(self):
        return getattr(self.args, 'results_dir', './result_data')

    def _build_model(self):
        if self.args.model not in self.MODEL_CLASSES:
            raise ValueError(f'Unknown model {self.args.model!r}. '
                             f'Available: {list(self.MODEL_CLASSES)}')
        model = self.MODEL_CLASSES[self.args.model].Model(self.args).float()

        # Optional transfer learning: warm-start from the weights of the last
        # trained instance of this architecture. Off by default so every base
        # station is trained from scratch (independent, comparable runs).
        if getattr(self.args, 'transfer', False):
            transfer_model_path = os.path.join(self._results_dir, f'transfer_model_{self.args.model}.pth')
            if os.path.exists(transfer_model_path):
                model.load_state_dict(torch.load(transfer_model_path))
                print("Loaded model for transfer learning from:", transfer_model_path)
            else:
                print("Transfer model file not found. Initializing new model instead.")

        return model

    def _model_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """Single dispatch point for all model forward calls.

        PatchTST / CATS / DeformableTST take only the input series.
        PerimidFormer additionally needs the time-feature marks; it returns a
        tensor of length label_len + pred_len (the caller slices to pred_len).
        """
        if self.args.model == 'PerimidFormer':
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return self.model(batch_x)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # AdamW + weight decay is the recipe DeformableTST was published with;
        # the other three models use plain Adam in their original repos.
        if getattr(self.args, 'optimizer', 'Adam') == 'AdamW':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,
                                      weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        batch_sizes = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
                batch_sizes.append(len(pred))
        # weight by batch size: with drop_last=False the last batch is
        # smaller, an unweighted mean would over-count its windows
        total_loss = np.average(total_loss, weights=batch_sizes)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        print('train data, train loader', len(train_data), len(train_loader))
        vali_data, vali_loader = self._get_data(flag='val')
        print('validation data, validation loader', len(vali_data), len(vali_loader))
        test_data, test_loader = self._get_data(flag='test')
        print('test data, test loader', len(test_data), len(test_loader))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # Exactly one LR schedule is active, selected by args.lradj:
        #   'TST'            -> OneCycleLR, stepped every iteration (PatchTST option)
        #   'cos'            -> linear warmup (args.warmup_epochs) + cosine annealing
        #                       (DeformableTST recipe)
        #   'type1/2/3', ... -> epoch-level decay via adjust_learning_rate
        onecycle_scheduler = None
        cosine_scheduler = None
        if self.args.lradj == 'TST':
            onecycle_scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                         steps_per_epoch=train_steps,
                                                         pct_start=self.args.pct_start,
                                                         epochs=self.args.train_epochs,
                                                         max_lr=self.args.learning_rate)
        elif self.args.lradj == 'cos':
            warmup = max(self.args.warmup_epochs, 0)
            cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=model_optim,
                                                              T_max=self.args.train_epochs - warmup)
            if warmup > 0:
                print('We will warm up', warmup, 'epochs!')

        for epoch in range(self.args.train_epochs):

            # warmup + cosine annealing, both applied at the start of the epoch
            # (same timing as the original DeformableTST implementation)
            if self.args.lradj == 'cos':
                if epoch < self.args.warmup_epochs:
                    warm_up(model_optim, epoch, self.args)
                else:
                    cosine_scheduler.step()
                    print(f'Updating learning rate to {cosine_scheduler.get_last_lr()[0]:.7f}')

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, onecycle_scheduler, epoch + 1, self.args, printout=False)
                    onecycle_scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj == 'TST':
                print('Updating learning rate to {}'.format(onecycle_scheduler.get_last_lr()[0]))
            elif self.args.lradj != 'cos':
                adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # transfer_save=False (federated mode): stations load the shared
        # global weights but never overwrite them
        if getattr(self.args, 'transfer', False) and getattr(self.args, 'transfer_save', True):
            transfer_model_path = os.path.join(self._results_dir, f'transfer_model_{self.args.model}.pth')
            torch.save(self.model.state_dict(), transfer_model_path)
            print(f"Model saved for transfer learning at {transfer_model_path}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = os.path.join(self._results_dir, 'test_results', setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        # concatenate over batches: with drop_last=False the final batch can
        # be smaller, so np.array() over the list would be ragged
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = os.path.join(self._results_dir, 'results', setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open(os.path.join(self._results_dir, 'result.txt'), 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, float(np.mean(corr))]))
        # full named metric set (sMAPE, WAPE, MASE, bias, peak/underprediction
        # diagnostics) for the analysis tooling; metrics.npy keeps its legacy
        # 7-slot layout for backward compatibility
        with open(folder_path + 'metrics_ext.json', 'w') as f:
            json.dump(metric_all(preds, trues), f, indent=2)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, :]

                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = os.path.join(self._results_dir, 'results', setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
