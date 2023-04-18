import numpy as np
from torch.utils.data import Dataset
import torch

class Cells(Dataset):
    """BBBC021 dataset."""

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