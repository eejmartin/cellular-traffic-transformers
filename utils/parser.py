import argparse

class Parser:

    def __init__(self, data, description):
        self.parser = argparse.ArgumentParser(description=description)

        # random seed (-s/--seed accepted here too so run.py's CLI arguments
        # do not error in argparse; run.py sets the effective value afterwards)
        self.parser.add_argument('-s', '--seed', '--random_seed', dest='random_seed',
                                 type=int, default=2021, help='random seed')

        # dataset selection for run.py (parsed here too so run.py's CLI
        # arguments do not error in argparse)
        self.parser.add_argument('-f', '--files', type=str, default='bs', choices=['bs', 'big_data'],
                                 help='which per-station dataset run.py trains on: '
                                      'bs = data/{big,medium,small}_bs, big_data = data/big_data/parsed/concat')
        self.parser.add_argument('-l', '--limit', type=int, default=None,
                                 help='train only the first N station files (useful for the 17k-station big_data set)')

        # basic config
        self.parser.add_argument('--is_training', type=int, default=1, help='status')
        self.parser.add_argument('--model_id', type=str, default='test', help='model id')
        self.parser.add_argument('--model', type=str, default='Autoformer',
                            help='model name, options: [Autoformer, Informer, Transformer]')
        
        # data loader
        self.parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
        self.parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        self.parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        self.parser.add_argument('--features', type=str, default='M',
                   help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        self.parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
        self.parser.add_argument('--freq', type=str, default='h',
                  help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        self.parser.add_argument('--checkpoints', type=str, default='./result_data/checkpoints/', help='location of model checkpoints')
        self.parser.add_argument('--results_dir', type=str, default='./result_data', help='root folder for results/test_results/result.txt of this run')

        # forecasting task
        self.parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        self.parser.add_argument('--label_len', type=int, default=48, help='start token length')
        self.parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

        self.parser.add_argument('--inverse',type=bool, default=False,help='use inverse transform')
        self.parser.add_argument('--ratios', type=str, default='0.7,0.1,0.2', help='train,validation,test ratios (comma-separated, must sum to 1)')
        self.parser.add_argument('--scale', type=bool, default=False, help='standardize features with a StandardScaler fit on the train split')
        # -t and the transfer-strategy options are accepted here too so
        # run.py's CLI arguments do not error in argparse (same reason as
        # -f / -l above); the effective values are the ones run.py sets on
        # the merged config afterwards
        self.parser.add_argument('-t', '--transfer', action='store_true', default=False,
                                 help='warm-start each station from the last saved weights of the same architecture')
        self.parser.add_argument('--transfer-mode', dest='transfer_mode',
                                 choices=['sequential', 'ordered', 'grouped', 'federated', 'combined'],
                                 default='sequential', help='transfer strategy (see run.py)')
        self.parser.add_argument('--groups', type=int, default=3,
                                 help='clusters for --transfer-mode grouped')
        self.parser.add_argument('--fed-rounds', dest='fed_rounds', type=int, default=5,
                                 help='FedAvg rounds for --transfer-mode federated')
        self.parser.add_argument('--fed-epochs', dest='fed_epochs', type=int, default=5,
                                 help='local epochs per FedAvg round')

        # DLinear
        #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

        # PatchTST
        self.parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
        self.parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
        self.parser.add_argument('--patch_len', type=int, default=16, help='patch length')
        self.parser.add_argument('--stride', type=int, default=8, help='stride')
        self.parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
        self.parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
        self.parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
        self.parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
        self.parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
        self.parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
        self.parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

        # Formers 
        self.parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
        self.parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
        self.parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
        self.parser.add_argument('--c_out', type=int, default=7, help='output size')
        self.parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        self.parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        self.parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        self.parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        self.parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        self.parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        self.parser.add_argument('--factor', type=int, default=1, help='attn factor')
        self.parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        self.parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        self.parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        self.parser.add_argument('--activation', type=str, default='gelu', help='activation')
        self.parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        self.parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

        # optimization
        self.parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        # self.parser.add_argument('--itr', type=int, default=2, help='experiments times')
        self.parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
        self.parser.add_argument('--batch_size', type=int, default=512, help='batch size of train input data')
        self.parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
        self.parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        self.parser.add_argument('--des', type=str, default='test', help='exp description')
        self.parser.add_argument('--loss', type=str, default='mse', help='loss function')
        self.parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
        self.parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
        self.parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU
        self.parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
        self.parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        self.parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
        self.parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

        # CATS
        self.parser.add_argument('--QAM_start', type=float, default=0.1, help='masking start probability')
        self.parser.add_argument('--QAM_end', type=float, default=0.3, help='masking end probability')
        # self.parser.add_argument('--patch_len', type=int, default=24, help='patch length')
        # self.parser.add_argument('--stride', type=int, default=24, help='stride')
        # self.parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
        # self.parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
        self.parser.add_argument('--query_independence', action='store_true', default=True, help='sharing query across dimension')
        self.parser.add_argument('--store_attn', action='store_true', default=False, help='store attention score')

        # DeformableTST
        self.parser.add_argument('--n_vars', type=int, default=321, help='number of variables in the input series')
        self.parser.add_argument('--revin_affine', type=int, default=0, help='use RevIN-affine; True 1 False 0')
        self.parser.add_argument('--revin_subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract the last value')
        self.parser.add_argument('--stem_ratio', type=int, default = 1, help='down sampling ratio in stem layer')
        self.parser.add_argument('--down_ratio', type=int, default = 2, help='down sampling ratio in DownSampling layer between two stages')
        self.parser.add_argument('--fmap_size', type=int, default = 768, help='feature series length')
        self.parser.add_argument('--dims', nargs='+', type=int, default=[32, 64, 128, 256], help='dims for each stage')
        self.parser.add_argument('--depths', nargs='+', type=int, default=[1, 1, 1, 1], help='number of Transformer blocks for each stage')
        self.parser.add_argument('--drop_path_rate', type=float, default = 0.3, help='drop path rate')
        self.parser.add_argument('--layer_scale_value', nargs='+', type=float, default=[-1, -1, -1, -1], help='layer_scale_init_value')
        self.parser.add_argument('--use_pe', nargs='+', type=int, default=[1,1,1,1], help='use pe; True 1 False 0')
        self.parser.add_argument('--use_lpu', nargs='+', type=int, default=[1,1,1,1], help='use Local Perception Unit; True 1 False 0')
        self.parser.add_argument('--local_kernel_size', nargs='+', type=int, default=[3, 3, 3, 3], help='kernel size for LPU')
        self.parser.add_argument('--expansion', type=int, default=1, help='ffn ratio')
        self.parser.add_argument('--drop', type=float, default = 0.0, help='dropout prob for FFN module')
        self.parser.add_argument('--use_dwc_mlp', nargs='+', type=int, default=[1,1,1,1], help='use FFN with a DWConv; True 1 False 0')
        self.parser.add_argument('--heads', nargs='+', type=int, default=[4, 8, 16, 32], help='number of heads')
        self.parser.add_argument('--attn_drop', type=float, default = 0.0, help='dropout prob for attention map in attention module')
        self.parser.add_argument('--proj_drop', type=float, default = 0.0, help='dropout prob for proj in attention module')
        self.parser.add_argument('--stage_spec', nargs='+', type=list, default=[['D'], ['D'], ['D','D','D'], ['D']], help='type of blocks in each stage')
        self.parser.add_argument('--window_size', nargs='+', type=int, default=[3, 3, 3, 3], help='kernel size for window attention')
        self.parser.add_argument('--nat_ksize', nargs='+', type=int, default=[3, 3, 3, 3], help='kernel size for neighborhood attention')
        self.parser.add_argument('--ksize', nargs='+', type=int, default=[9, 7, 5, 3], help='kernel size for offset sub-network')
        # self.parser.add_argument('--stride', nargs='+', type=int, default=[8, 4, 2, 1], help='stride for offset sub-network')
        self.parser.add_argument('--n_groups', nargs='+', type=int, default=[2, 4, 8, 16], help='number of offset groups')
        self.parser.add_argument('--offset_range_factor', nargs='+', type=float, default=[-1, -1, -1, -1], help='restrict the offset value in a small range')
        self.parser.add_argument('--no_off', nargs='+', type=int, default=[0,0,0,0], help='not use offset; True 1 False 0')
        self.parser.add_argument('--dwc_pe', nargs='+', type=int, default=[0,0,0,0], help='use DWC-pe; True 1 False 0')
        self.parser.add_argument('--fixed_pe', nargs='+', type=int, default=[0,0,0,0], help='use fixed pe; True 1 False 0')
        self.parser.add_argument('--log_cpb', nargs='+', type=int, default=[0,0,0,0], help='use pe of SWin-v2; True 1 False 0')
        # self.parser.add_argument('--batch_size', type=int, default=512, help='batch size')
        # self.parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
        self.parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')
        # self.parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        # self.parser.add_argument('--loss', type=str, default='MSE', help='loss function')
        self.parser.add_argument('--optimizer', type=str, default='AdamW', help='type of optimizer, choose from [AdamW, Adam]')
        self.parser.add_argument('--weight_decay', type=float, default=0.05, help='weight_decay')
        self.parser.add_argument('--head_type', type=str, default='Flatten', help='Flatten')
        self.parser.add_argument('--use_head_norm', type=int, default=1, help='use final LN layer; True 1 False 0')

        # PerimidFormer
        self.parser.add_argument('--task_name', type=str, default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        # forecasting task
        self.parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
        # inputation task
        self.parser.add_argument('--mask_rate', type=float, default=0.125, help='mask ratio')
        # anomaly detection task
        self.parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')
        # model define
        self.parser.add_argument('--top_k', type=int, default=2, help='num of most significant frequencies')
        self.parser.add_argument('--chan_in', type=int, default=21, help='input channel size')
        # self.parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        self.parser.add_argument('--feature_flows_dim', type=int, default=512, help='feature flows dim')
        # self.parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        self.parser.add_argument('--layers', type=int, default=1, help='num of model layers')


        self.args = self.parser.parse_args()

        for key, value in data.items():
            setattr(self.args, key, value)