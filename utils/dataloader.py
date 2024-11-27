import pickle
import numpy as np
import datetime
import scipy.sparse as sp
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from os.path import join as join


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMaxNormScaler():
    """
    min max normalization the input
    """

    def __init__(self, _min, _max):
        self._min = _min
        self._max = _max

    def transform(self, x):
        r"""
        Max-min normalization

        _max: float
            Max
        _min: float
            Min
        """
        x = 1. * (x - self._min) / (self._max - self._min)
        x = x * 2. - 1.
        return x

    def inverse_transform(self, x):
        r"""
        Max-min re-normalization

        _max: float
            Max
        _min: float
            Min
        """
        x = (x + 1.) / 2.
        x = 1. * x * (self._max - self._min) + self._min
        return x


class TrafficFlowDataset(Dataset):
    """
    Dataset for PEMS03, PEMS04, PEMS07, PEMS08, CA2019 to predict traffic flow
    """

    def __init__(self, args, mode):
        super().__init__()
        assert mode in ["train", "val", "test"]

        num_feat = 1  # flow
        pred_len = args.pred_len
        seq_len = args.seq_len
        # seq_len = 288
        add_time_of_day = args.tod
        add_day_of_week = args.dow
        train_ratio = 0.6
        val_ratio = 0.2
        data_file_path = join(args.dataset_dir, args.dataset, f'{args.dataset}.npz')
        steps_per_day = args.steps_per_day

        # read data
        data = np.load(data_file_path)["data"]
        data = data[..., 0:num_feat]

        l, n, f = data.shape
        num_samples = l - (seq_len + pred_len) + 1

        index_list = []
        for t in range(seq_len, num_samples + seq_len):
            index = (t - seq_len, t, t + pred_len)
            index_list.append(index)

        train_num_short = round(num_samples * train_ratio)
        val_num_short = round(num_samples * val_ratio)
        test_num_short = num_samples - train_num_short - val_num_short
        num_dict = {
            'train': train_num_short,
            'val': val_num_short,
            'test': test_num_short
        }
        print(f"number of {mode} samples:{num_dict[mode]}")

        train_index = index_list[:train_num_short]
        val_index = index_list[train_num_short: train_num_short + val_num_short]
        test_index = index_list[train_num_short +
                                val_num_short: train_num_short + val_num_short + test_num_short]

        data_train = data[:train_index[-1][1], ...]
        self.scaler = StandardScaler(mean=data_train[..., 0].mean(), std=data_train[..., 0].std())
        data = self.scaler.transform(data)

        # add external feature
        feature_list = [data]
        if add_time_of_day:
            # numerical time_of_day
            tod = [i % steps_per_day /
                   steps_per_day for i in range(data.shape[0])]
            tod = np.array(tod)
            tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
            feature_list.append(tod_tiled)

        if add_day_of_week:
            # numerical day_of_week
            dow = [(i // steps_per_day) % 7 / 7 for i in range(data.shape[0])]
            dow = np.array(dow)
            dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
            feature_list.append(dow_tiled)

        processed_data = np.concatenate(feature_list, axis=-1, dtype=np.float32)

        # dump data
        index = {}
        index["train"] = train_index
        index["val"] = val_index
        index["test"] = test_index

        self.data = processed_data
        self.index = index[mode]

    def __getitem__(self, index):
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        idx = list(self.index[index])
        if isinstance(idx[0], int):
            # continuous index
            history_data = self.data[idx[0]:idx[1]]
            future_data = self.data[idx[1]:idx[2]]
        else:
            # discontinuous index or custom index
            # NOTE: current time $t$ should not included in the index[0]
            history_index = idx[0]  # list
            assert idx[1] not in history_index, "current time t should not included in the idx[0]"
            history_index.append(idx[1])
            history_data = self.data[history_index]
            future_data = self.data[idx[1], idx[2]]

        return history_data, future_data

    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)


class DataLoaderCalifornia(object):
    def __init__(self, data, batch_size, input_length, output_length):
        self.seq_length_x = input_length
        self.seq_length_y = output_length
        self.y_start = 1
        self.batch_size = batch_size
        self.current_ind = 0
        self.x_offsets = np.sort(np.concatenate((np.arange(-(self.seq_length_x - 1), 1, 1),)))
        self.y_offsets = np.sort(np.arange(self.y_start, (self.seq_length_y + 1), 1))
        self.min_t = abs(min(self.x_offsets))
        self.max_t = abs(data.shape[0] - abs(max(self.y_offsets)))
        mod = (self.max_t - self.min_t) % batch_size
        if mod != 0:
            self.data = data[:-mod]
        else:
            self.data = data
        self.max_t = abs(self.data.shape[0] - abs(max(self.y_offsets)))
        self.permutation = [i for i in range(self.min_t, self.max_t)]
        self.size = len(data)
        self.num_batch = int(self.size // self.batch_size)

    def shuffle(self):
        self.permutation = np.random.permutation([i for i in range(self.min_t, self.max_t)])

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < len(self.permutation):
                if self.batch_size > 1:
                    x_batch = []
                    y_batch = []
                    for i in range(self.batch_size):
                        x_i = self.data[self.permutation[self.current_ind + i] + self.x_offsets, ...]
                        y_i = self.data[self.permutation[self.current_ind + i] + self.y_offsets, ...]
                        x_batch.append(x_i)
                        y_batch.append(y_i)

                    x_batch = np.stack(x_batch, axis=0)
                    y_batch = np.stack(y_batch, axis=0)
                else:
                    x_batch = self.data[self.permutation[self.current_ind] + self.x_offsets, ...]
                    y_batch = self.data[self.permutation[self.current_ind] + self.y_offsets, ...]
                    x_batch = np.expand_dims(x_batch, axis=0)
                    y_batch = np.expand_dims(y_batch, axis=0)
                yield (x_batch, y_batch)
                self.current_ind += self.batch_size

        return _wrapper()

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self.get_iterator()


def get_dataloader(args):
    if args.dataset == 'California':
        data = {}
        for category in ['train', 'val', 'test']:
            data[category] = np.load(join(args.dataset_dir, args.dataset, category + '.npy'))
            data[category][..., 2] = data[category][..., 2] / 7
            print('*' * 10, category, data[category].shape, '*' * 10)
        scaler = StandardScaler(mean=data['train'][..., 0].mean(), std=data['train'][..., 0].std())
        # Data format
        for category in ['train', 'val', 'test']:
            data[category][..., 0] = scaler.transform(data[category][..., 0])
        data['train_loader'] = DataLoaderCalifornia(data['train'], args.batch_size, args.seq_len, args.pred_len)
        data['val_loader'] = DataLoaderCalifornia(data['val'], args.batch_size, args.seq_len, args.pred_len)
        data['test_loader'] = DataLoaderCalifornia(data['test'], args.batch_size, args.seq_len, args.pred_len)
        data['scaler'] = scaler
        adj_file_path = join(args.dataset_dir, args.dataset, 'adj_mx_distance.pkl')
        adj_mx = load_processed_adj(adj_file_path)
        return data['train_loader'], data['val_loader'], data['test_loader'], scaler, adj_mx

    if args.dataset in ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'CA2019']:
        train_dataset = TrafficFlowDataset(args, 'train')
        val_dataset = TrafficFlowDataset(args, 'val')
        test_dataset = TrafficFlowDataset(args, 'test')
        scaler = train_dataset.scaler
        # load adj
        adj_file_path = join(args.dataset_dir, args.dataset, 'adj_mx_distance.pkl')
        adj_mx = load_processed_adj(adj_file_path)
    else:
        raise NotImplementedError

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)

    return train_loader, val_loader, test_loader, scaler, adj_mx


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    try:
        _, _, adj_mx = load_pickle(pkl_filename)
    except:
        adj_mx = load_pickle(pkl_filename)

    return adj_mx


def load_processed_adj(adj_file):
    adj_mx = load_adj(adj_file)
    adj_mx = asym_adj(adj_mx)
    return adj_mx
