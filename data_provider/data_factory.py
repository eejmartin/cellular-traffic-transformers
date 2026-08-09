#from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Custom, Dataset_Pred

data_dict = {
    #'ETTh1': Dataset_ETT_hour,
    #'ETTh2': Dataset_ETT_hour,
    #'ETTm1': Dataset_ETT_minute,
    #'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        # never drop the tail batch at evaluation time: with batch_size=6 a
        # drop_last=True test loader silently discards up to 5 of the last
        # test windows (e.g. 9 -> 6 on the 48h big_data series), corrupting
        # every reported metric
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    elif flag == 'val':
        # deterministic full-coverage validation: with shuffle+drop_last the
        # early-stopping criterion saw a different random subset of the val
        # windows every epoch (e.g. 18 of 20), making model selection noisy
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

        #print('root_path is ', args.root_path)
        #print('data_path is ', args.data_path)
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=getattr(args, 'scale', False),
        timeenc=timeenc,
        freq=freq,
        ratios=args.ratios
    )
    
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader