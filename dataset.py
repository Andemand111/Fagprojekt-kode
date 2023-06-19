import numpy as np
from torch.utils.data import Dataset
import torch

"""

Various Dataset classes for different uses
In short:
    SmallCells is for a subset of cells contained in a matrix
    Cells is for all files in a folder.
    Encodings is for the latent encodings for downstream classification task
    
"""


class SmallCells(Dataset):
    """
    
    Class for small dataset
    Input:
        data : a datamatrix containing cell images
        channel : which colorchannel to return
        device : a torch device
    
    """
    
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
    """" 
    
    Class for big dataset
    Input:
        path : string, where are the files located
        channel : which colorchannel to return
        device : a torch device
    
    """

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
    """" 
    
    Class for encodings
    Input:
        encodings : datamatrix containing the encodings
        labels : array of labels
        indxs : array of indxs, if not all encodings have labels
        device : a torch device
    
    """
    
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