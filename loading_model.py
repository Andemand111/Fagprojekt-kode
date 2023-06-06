from dataset import SmallCells
from models import VAE
from graphics import Graphics

import numpy as np
import torch

latent_size = 700

data = np.load("C:/Users/gusta/OneDrive/Skrivebord/KI & Data\Semester 4\Fagprojekt\Data\celle_data_20000.npy")
train_data = SmallCells(data)
dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=128,
    drop_last=True, shuffle=True)

vae = VAE(latent_size)
plots = Graphics(vae, train_data)

vae.load_model("rgb_model_normal")

#%%
stats = np.load("rgb_model_normal_stats.npy")
vae.stats = stats
plots.show_convergence()
plots.random_generations()
plots.reconstructions()
plots.interpolate([543,432])