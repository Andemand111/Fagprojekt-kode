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

        sample = torch.from_numpy(sample).flatten().float()

        if self.device is not None:
            sample = sample.to(self.device)

        return sample

### defines the dataset class and makes an instance of it
class Encodings(Dataset):
    def __init__(self, encodings, labels, indxs, device):
        self.labels = labels
        self.indxs = indxs
        self.device = device
        self.encodings = encodings
        self.encodings_with_labels = encodings[indxs]
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.encodings_with_labels[idx].to(self.device)
        y = torch.tensor(self.labels[idx]).to(self.device)
        return X, y