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

        sample = torch.from_numpy(sample).flatten().float()


        if self.device is not None:
            sample = sample.to(self.device)

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
            
        sample = torch.from_numpy(sample).flatten().float()

        if self.device is not None:
            sample = sample.to(self.device)

        return sample


class ClassifyCells(Dataset):

    def __init__(self, path, models, color_channels, indxs, labels):
        self.path = path
        self.models = models
        self.color_channels = color_channels
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
        
        X = torch.tensor([])        
        for model, channel in zip(self.models, self.color_channels):
            if channel is not None:
                encoding = model.encode(sample[:, :, channel])
            else:
                encoding  = model.encode(sample)
            
            X = torch.cat((X, encoding.flatten()))

        return X, y
