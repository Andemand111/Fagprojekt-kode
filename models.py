import torch
import numpy as np
from torch import log, lgamma
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

class Encoder(torch.nn.Module):
    """  Encoder class for VAE

    Arguments:
    latent_size : integer, the amount of latent variables in bottleneck

    Functions:
        forward:
            arguments:
                x :             a data matrix (n x 13872)

            returns: 
                mu, log_var:    the location and log scale of the distribution
                                of the latent variables

        """

    def __init__(self, latent_size, num_channels = 3):
        super(Encoder, self).__init__()

        self.latent_size = latent_size
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.lin1 = nn.Linear(10368, 2048)

        self.mu_lin = nn.Linear(2048, latent_size)
        self.log_var_lin = nn.Linear(2048, latent_size)

    def forward(self, x):
        x_reshaped = x.view(-1, 68, 68, self.num_channels).permute(0, 3, 1, 2)
        
        z = F.relu(self.conv1(x_reshaped))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = torch.flatten(z, start_dim=1)
        z = F.relu(self.lin1(z))

        mu = self.mu_lin(z)
        log_var = self.log_var_lin(z)

        return mu, log_var


class Decoder(nn.Module):
    """  Decoder class for VAE

    Arguments:
    latent_size : integer, the amount of latent variables in bottleneck

    Functions:
        forward:
            arguments:
                z :             a matrix of latent variables (n x latent_size)

            returns: 
                x_hat:          a data matrix (n x 13872); the reconstruced image 
                                given z (or rather a distribution parameter for each 
                                pixel value)

    """

    def __init__(self, latent_size, num_channels = 3):
        super(Decoder, self).__init__()

        self.lin1 = nn.Linear(latent_size, 2048)
        self.lin2 = nn.Linear(2048, 4096)
        self.lin3 = nn.Linear(4096, 68 * 68 * num_channels)

    def forward(self, z):
        x_hat = F.relu(self.lin1(z))
        x_hat = F.relu(self.lin2(x_hat))
        x_hat = torch.sigmoid(self.lin3(x_hat))
        return x_hat


class VAE(nn.Module):
    def __init__(self, latent_size, num_channels = 3):
        super(VAE, self).__init__()

        self.latent_size = latent_size
        self.num_channels = num_channels
        self.stats = None

        self.encoder = Encoder(latent_size, num_channels = num_channels)
        self.decoder = Decoder(latent_size, num_channels = num_channels)

    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decoder(sampled_z)

        return mu, log_var, x_hat

    def reconstruction_loss(self, x_hat, x_recon, distribution="beta", sigma=1):
        if distribution == "beta":
            eps = 1e-6
            x_recon = torch.clamp(x_recon, eps, 1-eps)
            x_hat = torch.clamp(x_hat, eps, 1 - eps)

            kappa = 4  # må ikke være mindre end eller lig med - 2
            alfa = x_hat * (kappa - 2) + 1
            beta = (1 - x_hat) * (kappa - 2) + 1

            ln_B = lgamma(alfa) + lgamma(beta) - \
                lgamma(torch.tensor(kappa))

            Re = -((alfa - 1) * log(x_recon) +
                   (beta - 1) * log(1 - x_recon) -
                   ln_B).sum(1).mean()

        elif distribution == "normal":
            n = x_hat.shape[1]
            var = sigma ** 2
            mse = ((x_hat - x_recon) ** 2).sum(1).mean() / var
            const = n * log(torch.tensor(2 * torch.pi * var))
            Re = 0.5 * (const + mse)

        return Re

    def train(self, num_epochs, dataloader, kl_beta=1, verbose=2, distribution="beta", sigma=1, callback=None, callback_args={}):
        self.stats = np.zeros((num_epochs, 3))
        optimizer = torch.optim.Adam(self.parameters())

        # for debugging
        torch.set_anomaly_enabled(True)

        # Training Loop
        for epoch in range(num_epochs):
            curr_stats = np.zeros((len(dataloader), 3))

            for i, x_recon in enumerate(dataloader):
                if verbose == 2:
                    print(f'batch number {i+1} out of {len(dataloader)}')

                optimizer.zero_grad()
                mu, log_var, x_hat = self(x_recon)

                Re = self.reconstruction_loss(
                    x_hat, x_recon, distribution, sigma)
                kl = 0.5 * (log_var.exp() + mu ** 2 -
                            1 - log_var).sum(1).mean()

                loss = Re + kl_beta * kl
                
                loss.backward()
                optimizer.step()

                curr_stats[i, :] = [loss.item(), Re.item(), kl.item()]

            epoch_stats = np.mean(curr_stats, axis=0)
            self.stats[epoch, :] = epoch_stats

            if verbose:
                print(f"Beta = {kl_beta}     epoch = {epoch + 1}")
                print(*epoch_stats, "\n")

            if callback is not None:
                callback(*callback_args)

        return self.stats

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("Model saved!")

    def load_model(self, filename, device="cpu"):
        if device=="cpu":    
            self.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(filename))
        print("Model loaded!")

    def encode(self, x):
        mu, log_var = self.encoder(x)

        return mu.detach()

    def decode(self, z):
        x_hat = self.decoder(z)

        return x_hat.detach()

class ClassifyNN(nn.Module):

    def __init__(self, num_features):
        super(ClassifyNN, self).__init__()
        
        self.lin1 = nn.Linear(num_features, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 13)

    def forward(self, x):
        y = F.relu(self.lin1(x))
        y = F.relu(self.lin2(y))
        y = F.softmax(self.lin3(y), 1)
        return y

    def train(self, num_epochs, dataloader, verbose=2):
        self.stats = np.zeros(num_epochs)
        optimizer = torch.optim.Adam(self.parameters())
        
        for epoch in range(num_epochs):
            curr_stats = np.zeros(len(dataloader))
            
            for i, (X, y) in enumerate(dataloader):                
                preds = self(X)
                y = F.one_hot(y.long(), 13).float()
                loss = F.binary_cross_entropy(preds, y)
                curr_stats[i] = loss.item()

                loss.backward()
                optimizer.step()
                
                if verbose == 2:
                    print(f"Batch {i + 1} out of {len(dataloader)}")
            
            epoch_stats = np.mean(curr_stats)
            self.stats[epoch] = epoch_stats
            
            if verbose:
                print(f"Epoch: {epoch + 1} / {num_epochs}")
                print(f"loss: {epoch_stats} \n")
                plt.plot(self.stats)
                plt.show()

    def test(self, data):
        pbar = tqdm(total = len(data))
        res = torch.zeros(len(data))
        
        print("testing classification model..")
        
        for i, (X, y) in enumerate(data):
            pred = self(X.view(1, -1))
            pred = pred.argmax(1).item()
            
            res[i] = (pred == y).astype(np.int64)
            pbar.update(1)
            
        print("... done!")
        
        accuracy = res.mean()
        return accuracy
                
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("Model saved!")
        
