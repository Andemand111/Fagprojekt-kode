from dataset import Cells
from models import Encoder, Decoder, VAE
from graphics import Graphics

import numpy as np
import torch

latent_size = 500

data = np.load("celle_data.npy")
train_data = Cells(data)
dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=128,
    drop_last=True, shuffle=True)

encoder = Encoder(latent_size)
decoder = Decoder(latent_size)

vae = VAE(encoder, decoder)
plots = Graphics(vae, train_data)

def callback():
    plots.random_generations()
    plots.reconstructions()
    plots.show_convergence()

stats = vae.train(60, dataloader, kl_beta=1, verbose=1, distribution="normal", callback=callback)

plots.interpolate()
plots.investigate_feature()