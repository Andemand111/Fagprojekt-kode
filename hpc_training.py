from models import Encoder, Decoder, VAE
from dataset import Cells
import torch

latent_size = 500
train_data = Cells()
    
dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=128,
    drop_last=True, shuffle=True)

encoder = Encoder(latent_size)
decoder = Decoder(latent_size)

vae = VAE(encoder, decoder)

stats = vae.train(60, dataloader, kl_beta=3)
vae.save_model("betavae")