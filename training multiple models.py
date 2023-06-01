from dataset import SmallCells
from models import VAE
from torch.utils.data import DataLoader

import numpy as np

big_latent_size = 800
small_latent_size = 100

data = np.load("celle_data.npy")

rgb_data = SmallCells(data)
r_data = SmallCells(data, channel=0)
g_data = SmallCells(data, channel=1)
b_data = SmallCells(data, channel=2)

args = dict(batch_size=128, drop_last=True, shuffle=True)
rgb_dataloader = dataloader = DataLoader(rgb_data, **args)
r_dataloader = dataloader = DataLoader(r_data, **args)
g_dataloader = dataloader = DataLoader(g_data, **args)
b_dataloader = dataloader = DataLoader(b_data, **args)

rgb_model = VAE(big_latent_size)
r_model = VAE(small_latent_size, num_channels=1)
g_model = VAE(small_latent_size, num_channels=1)
b_model = VAE(small_latent_size, num_channels=1)

rgb_stats = rgb_model.train(45, rgb_dataloader, kl_beta = 3)
r_stats = r_model.train(45, r_dataloader, kl_beta = 3)
g_stats = g_model.train(45, g_dataloader, kl_beta = 3)
b_stats = b_model.train(45, b_dataloader, kl_beta = 3)