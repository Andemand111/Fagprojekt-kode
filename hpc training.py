from models import Encoder, Decoder, VAE
import numpy as np
import torch

latent_size = 500

class Cells():
    def __init__(self):
        self.path = "/zhome/5a/2/167858/Desktop/merged_files/"

    def __len__(self):
        return 488000

    
    def __getitem__(self, idx):
        sample = np.load(self.path + str(idx)).astype(np.float32)
        
        # divide by max value in each channel
        sample = sample.reshape(-1,3)
        RGBmax = np.max(sample,axis=0)
        sample /= RGBmax
        
        sample = torch.from_numpy(sample).flatten()
        return sample
    
train_data = Cells()
    
dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=128,
    drop_last=True, shuffle=True)

encoder = Encoder(latent_size)
decoder = Decoder(latent_size)

vae = VAE(encoder, decoder)

stats = vae.train(60, dataloader, kl_beta=3, verbose = 1)
vae.save_model("betavae")