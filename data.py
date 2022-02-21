from re import X
import scipy.io
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.preprocessing import OrdinalEncoder

class dataset(Dataset):
    def __init__(self, X, y, train=True):
        self.X = X
        self.y = y
        self.train=train

    def __len__(self):
        return len(self.y)

    # def __getitem__(self, idx):
    #     rng = np.random.randint(0, high=200)
    #     if self.train:
    #         x = self.X[idx][:, rng:rng + 600]
    #     else:
    #         x = self.X[idx][:, 200: 800]
    #     return x, self.y[idx]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.train:
#             rn = np.random.randint(0, high=500)
#             x = x[:, rn:rn+4000]
            x = x[:, 0:4000]
        else:
            x = x[:, 0:4000]
        return x, self.y[idx]

def generate_x_train(mat):
    # out: num_data_points * chl * trial_length
    data = []
    last_label = False
    for i in range(0, len(mat['mrk'][0][0][0][0])-1):
        start_idx = mat['mrk'][0][0][0][0][i]
        end_idx = mat['mrk'][0][0][0][0][i+1]
        # to resolve shape issues, we use a shifted window 
        # (possible overlapping but acceptable given it's trivial)
        end_idx += (8000 + start_idx - end_idx) 
        data.append(mat['cnt'][start_idx: end_idx,].T)
    # add the last datapoint
    if len(mat['cnt']) - mat['mrk'][0][0][0][0][-1] >= 8000:
        last_label = True
        start_idx = mat['mrk'][0][0][0][0][-1]
        end_idx = start_idx + 8000
        data.append(mat['cnt'][start_idx: end_idx,].T)
    return np.array(data), last_label

def generate_y_train(mat, last_label):
    # out: 1 * num_labels
    class1, class2 = mat['nfo']['classes'][0][0][0][0][0], mat['nfo']['classes'][0][0][0][1][0]
    mapping = {-1: class1, 1: class2}
    labels = np.vectorize(mapping.get)(mat['mrk'][0][0][1])[0]
    if not last_label:
        labels = labels[:-1]
    return labels

def generate_data(files):
    X, y = [], []
    for file in files:
        print(file)
        mat = scipy.io.loadmat(file)
        X_batch, last_label = generate_x_train(mat)
        X.append(X_batch)
        y.append(generate_y_train(mat, last_label))
    X, y = np.concatenate(X, axis=0), np.concatenate(y)
    y = OrdinalEncoder().fit_transform(y.reshape(-1, 1))
    return X, y

def split_data(X, y):
    def get_idx():
        np.random.seed(seed=42)
        rng = np.random.choice(len(y), len(y), replace=False)
        return rng
    train_size, val_size, test_size = 1000, 197, 200
    indices = get_idx()
    train_idx, val_idx, test_idx = indices[0: train_size], \
                indices[train_size: train_size + val_size], indices[train_size + val_size:]
    train_X, train_y, val_X, val_y, test_X, test_y = \
        X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]
    return train_X, train_y, val_X, val_y, test_X, test_y

def get_loaders(train_X, train_y, val_X, val_y, test_X, test_y):
    train_set, val_set, test_set = dataset(train_X, train_y, True), dataset(val_X, val_y, False), dataset(test_X, test_y, False)
    data_loader_train = torch.utils.data.DataLoader(
        train_set, 
        batch_size=1, 
        num_workers=1,
        pin_memory=True, 
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
            val_set, 
            batch_size=1, 
            num_workers=1,
            pin_memory=True, 
            drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
            test_set, 
            batch_size=1, 
            num_workers=1,
            pin_memory=True, 
            drop_last=False,
    )
    dataloaders = {
        'train': data_loader_train,
        'val': data_loader_val,
        'test': data_loader_test
    }
    return dataloaders
