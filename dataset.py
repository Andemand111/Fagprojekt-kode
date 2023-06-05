import numpy as np
from torch.utils.data import Dataset
import torch


class SmallCells(Dataset):
    """ Takes a small dataset as input """

    def __init__(self, data, channel=None, device=None):
        self.data = data
        self.channel = channel
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = sample.reshape(68, 68, 3)

        # divide by max value in each channel
        sample /= np.amax(sample, axis=(0, 1))

        if self.channel is not None:
            sample = sample[:, :, self.channel]

        if self.device is not None:
            sample = sample.to(self.device)

        sample = torch.from_numpy(sample).flatten().float()
        return sample


class Cells(Dataset):
    """" Can be used to run on hpc """

    def __init__(self, path, channel=None, device=None):
        self.path = path
        self.channel = channel
        self.device = device

    def __len__(self):
        return 488000

    def __getitem__(self, idx):
        sample = np.load(self.path + str(idx)).astype(np.float32)
        sample = sample.reshape(68, 68, 3)

        # divide by max value in each channel
        sample /= np.amax(sample, axis=(0, 1))

        if self.channel is not None:
            sample = sample[:, :, self.channel]

        if self.device is not None:
            sample = sample.to(self.device)

        sample = torch.from_numpy(sample).flatten().float()
        return sample


class ClassifyCells(Dataset):

    def __init__(self, path, models, indxs, labels, channel=None):
        self.path = path
        self.models = models
        self.indxs = indxs
        self.labels = labels

    def __len__(self):
        return len(self.indxs)

    def __getitem__(self, idx):
        directory_idx = self.indxs[idx]

        y = self.labels[idx]
        sample = np.load(self.path + str(directory_idx)).astype(np.float32)

        # divide by max value in each channel
        sample /= np.amax(sample, axis=(0, 1))
        sample = torch.from_numpy(sample).flatten().float()

        X = torch.zeros(1000)
        X[0:700]    = self.models[0].encode(sample).flatten()
        X[700:800]  = self.models[1].encode(sample[:, :, 0]).flatten()
        X[800:900]  = self.models[2].encode(sample[:, :, 1]).flatten()
        X[900:1000] = self.models[3].encode(sample[:, :, 2]).flatten()

        return X, y
