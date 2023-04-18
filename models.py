import torch
import numpy as np
import torch.optim as optim
from torch import log, lgamma
import torch.nn as nn


class Encoder(torch.nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()

        self.latent_size = latent_size

        encoder_network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Flatten(),

            nn.Linear(10368, 2048),
            nn.LeakyReLU(),

            nn.Linear(2048, 2 * latent_size)
        )

        self.network = encoder_network

    def forward(self, x):
        x_reshaped = x.view(-1, 68, 68, 3).permute(0, 3, 1, 2)
        z = self.network(x_reshaped)
        mu, log_var = torch.chunk(z, 2, 1)

        return mu, log_var


class Decoder(torch.nn.Module):
    def __init__(self, latent_size):
        super(Decoder, self).__init__()

        decoder_network = nn.Sequential(
            nn.Linear(latent_size, 2048),
            nn.LeakyReLU(),

            nn.Linear(2048, 4096),
            nn.LeakyReLU(),

            nn.Linear(4096, 68 * 68 * 3),
            nn.Sigmoid(),
        )

        self.network = decoder_network

    def forward(self, z):
        x_hat = self.network(z)
        return x_hat


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, kl_beta=1):
        super(VAE, self).__init__()

        self.kl_beta = kl_beta
        self.latent_size = encoder.latent_size

        self.encoder = encoder
        self.decoder = decoder

    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps_ = torch.randn_like(std)

        return eps_.mul(std).add_(mu)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decoder(sampled_z)

        return mu, log_var, x_hat

    def train(self, num_epochs, dataloader, verbose=1, learning_rate=0.001, reconstruction="beta", callback=None):
        self.stats = np.zeros((num_epochs, 3))
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # for debugging
        torch.set_anomaly_enabled(True)
        eps = 1e-6

        # Training Loop
        for epoch in range(num_epochs):
            curr_stats = np.zeros((len(dataloader), 3))

            for i, x_recon in enumerate(dataloader):
                if verbose == 2:
                    print(f'batch number {i+1} out of {len(dataloader)}')

                optimizer.zero_grad()
                mu, log_var, x_hat = self(x_recon)

                x_recon = torch.clamp(x_recon, eps, 1-eps)
                x_hat = torch.clamp(x_hat, eps, 1 - eps)

                precision = 100
                alfa = precision * x_hat
                beta = precision * (1 - x_hat)

                if reconstruction == "beta":
                    ln_B = lgamma(alfa) + lgamma(beta) - \
                        lgamma(torch.tensor(precision))
                    Re = -((alfa - 1) * log(x_recon) +
                           (beta - 1) * log(1 - x_recon)
                           - ln_B).sum(1).mean()

                elif reconstruction == "normal":
                    Re = torch.pow(x_hat - x_recon, 2).sum(1).mean()

                kl = (-0.5 * (1 + log_var - mu ** 2 - log_var.exp())).sum(1).mean()

                loss = Re + self.kl_beta * kl
                loss.backward()
                optimizer.step()

                curr_stats[i, :] = [loss.item(), Re.item(), kl.item()]

            epoch_stats = np.mean(curr_stats, axis=0)
            self.stats[epoch, :] = epoch_stats

            if verbose:
                print(f"Beta = {self.kl_beta}     epoch = {epoch}")
                print(*epoch_stats, "\n")

            if callback is not None:
                callback()

        return self.stats

    def encode(self, x):
        mu, log_var = self.encoder(x)

        return mu.detach()

    def decode(self, z):
        x_hat = self.decoder(z)

        return x_hat.detach()
