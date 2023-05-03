from dataset import SmallCells
from models import Encoder, Decoder, VAE
from graphics import Graphics

import numpy as np
import torch

latent_size = 500

data = np.load("celle_data.npy")
train_data = SmallCells(data)
dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=128,
    drop_last=True, shuffle=True)

encoder = Encoder(latent_size)
decoder = Decoder(latent_size)

vae = VAE(encoder, decoder)
plots = Graphics(vae, train_data)

vae.load_model("betavae")