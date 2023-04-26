from dataset import Cells
from models import Encoder, Decoder, VAE
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

stats = vae.train(60, dataloader, kl_beta=3, verbose = 1)
vae.save_model("betavae")