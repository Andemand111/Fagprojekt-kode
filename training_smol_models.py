from dataset import Cells
from models import VAE
from torch.utils.data import DataLoader

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

small_latent_size = 100

path = "/zhome/5a/2/167858/Desktop/merged_files/" # path to data

r_data = Cells(path=path, channel=0, device=device)
g_data = Cells(path=path, channel=1, device=device)
b_data = Cells(path=path, channel=2, device=device)

args = dict(batch_size=128, drop_last=True, shuffle=True)
r_dataloader = dataloader = DataLoader(r_data, **args)
g_dataloader = dataloader = DataLoader(g_data, **args)
b_dataloader = dataloader = DataLoader(b_data, **args)

r_model = VAE(small_latent_size, num_channels=1).to(device)
r_model.save_model("r_model")
r_stats = r_model.train(45, r_dataloader, kl_beta = 3, verbose=1)


g_model = VAE(small_latent_size, num_channels=1).to(device)
g_stats = g_model.train(45, g_dataloader, kl_beta = 3, verbose=1)
g_model.save_model("g_model")


b_model = VAE(small_latent_size, num_channels=1).to(device)
b_stats = b_model.train(45, b_dataloader, kl_beta = 3, verbose=1)
b_model.save_model("b_model")

np.save("r_model_stats", r_stats)
np.save("g_model_stats", g_stats)
np.save("b_model_stats", b_stats)