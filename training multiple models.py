from dataset import SmallCells
from models import VAE
from torch.utils.data import DataLoader

import numpy as np
import torch

device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")

big_latent_size = 700
small_latent_size = 100

data = np.load("celle_data.npy")

rgb_data = SmallCells(data, device=device)
r_data = SmallCells(data, channel=0, device=device)
g_data = SmallCells(data, channel=1, device=device)
b_data = SmallCells(data, channel=2, device=device)

args = dict(batch_size=128, drop_last=True, shuffle=True)
rgb_dataloader = dataloader = DataLoader(rgb_data, **args)
r_dataloader = dataloader = DataLoader(r_data, **args)
g_dataloader = dataloader = DataLoader(g_data, **args)
b_dataloader = dataloader = DataLoader(b_data, **args)

rgb_model = VAE(big_latent_size)
r_model = VAE(small_latent_size, num_channels=1).to(device)
g_model = VAE(small_latent_size, num_channels=1).to(device)
b_model = VAE(small_latent_size, num_channels=1).to(device)

rgb_stats = rgb_model.train(45, rgb_dataloader, kl_beta = 3, verbose=1)
r_stats = r_model.train(45, r_dataloader, kl_beta = 3, verbose=1)
g_stats = g_model.train(45, g_dataloader, kl_beta = 3, verbose=1)
b_stats = b_model.train(45, b_dataloader, kl_beta = 3, verbose=1)

rgb_model.save_model("rgb_model")
r_model.save_model("r_model")
g_model.save_model("g_model")
b_model.save_model("b_model")

np.save("rgb_model_stats", rgb_stats)
np.save("r_model_stats", r_stats)
np.save("g_model_stats", g_stats)
np.save("b_model_stats", b_stats)