from dataset import Cells
from models import VAE
from torch.utils.data import DataLoader

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

big_latent_size = 700

path = "/zhome/5a/2/167858/Desktop/merged_files/" # path to data

rgb_data = Cells(path=path, device=device)


args = dict(batch_size=128, drop_last=True, shuffle=True)
rgb_dataloader = DataLoader(rgb_data, **args)

rgb_model = VAE(big_latent_size).to(device)

rgb_stats = rgb_model.train(45, rgb_dataloader, kl_beta = 3, verbose=1)

rgb_model.save_model("rgb_model")

np.save("rgb_model_stats", rgb_stats)
