import numpy as np
from torch.utils.data import Dataset
import torch


class SmallCells(Dataset):
    """ Takes a small dataset as input """

    def __init__(self, data, channel=None):
        self.data = data
        self.channel = channel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = sample.reshape(68,68,3)

        # divide by max value in each channel
        sample /= np.amax(sample, axis=(0, 1))
        
        if self.channel is not None:
            sample = sample[:, :, self.channel]
        
        sample = torch.from_numpy(sample).flatten().float()
        return sample


class Cells(Dataset):
    """" Can be used to run on hpc """

    def __init__(self, path, channel=None):
        self.path = path
        self.channel = channel

    def __len__(self):
        return 488000

    def __getitem__(self, idx):
        sample = np.load(self.path + str(idx)).astype(np.float32)
        sample = sample.reshape(68,68,3)

        # divide by max value in each channel
        sample /= np.amax(sample, axis=(0, 1))
        
        if self.channel is not None:
            sample = sample[:, :, self.channel]
            
        sample = torch.from_numpy(sample).flatten()
        return sample


class ClassifyCells(Dataset):

    def __init__(self, path, model, indxs, labels, channel=None):
        mask = indxs <= 146108
        self.path = path
        self.model = model
        self.indxs = indxs[mask]
        self.labels = labels[mask]

    def __len__(self):
        return len(self.indxs)

    def __getitem__(self, idx):
        directory_idx = self.indxs[idx]
        try:
            y = self.labels[idx]
            sample = np.load(self.path + str(directory_idx)).astype(np.float32)
        except:
            y = self.labels[0]
            sample = np.load(self.path + str(0)).astype(np.float32)
    
        # divide by max value in each channel
        sample /= np.amax(sample, axis=(0, 1))
        sample = torch.from_numpy(sample).flatten()
        X = self.model.encode(sample).flatten()

        return X, y
