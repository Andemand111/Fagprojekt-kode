import numpy as np
from torch.utils.data import Dataset
import torch

class SmallCells(Dataset):
    """ Takes a small dataset as input """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # divide by max value in each channel
        sample = sample.reshape(-1,3)
        RGBmax = np.max(sample,axis=0)
        sample /= RGBmax
        
        sample = torch.from_numpy(sample).flatten().float()
        return sample
    
class Cells(Dataset):
    """" Can be used to run on hpc """
    
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return 488000

    
    def __getitem__(self, idx):
        sample = np.load(self.path + str(idx)).astype(np.float32)
        
        # divide by max value in each channel
        sample = sample.reshape(-1,3)
        RGBmax = np.max(sample,axis=0)
        sample /= RGBmax
        
        sample = torch.from_numpy(sample).flatten()
        return sample