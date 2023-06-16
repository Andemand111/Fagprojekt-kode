import torch
import numpy as np
from torch import log, lgamma
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

def calculate_conv_size(h, w, kernel, padding, stride, iterations):
    new_h = np.floor((h - kernel+ 2 * padding) / stride) + 1
    new_w = np.floor((w - kernel + 2 * padding) / stride) + 1
    while iterations > 1:
        return calculate_conv_size(new_h, new_w, kernel, padding, stride, iterations - 1)
    
    return new_h, new_w

class Encoder(torch.nn.Module):

    def __init__(self, latent_size, height = 68, width = 68, num_channels = 3):
        super(Encoder, self).__init__()

        self.latent_size = latent_size
        self.height = height
        self.width = width
        self.num_channels = num_channels
        
        conv_args = dict(kernel_size=(3,3), stride=2, padding=1)
        
        self.conv1 = nn.Conv2d(num_channels, 32, **conv_args)
        self.conv2 = nn.Conv2d(32, 64, **conv_args)
        self.conv3 = nn.Conv2d(64, 128, **conv_args)
        
        new_h, new_w = calculate_conv_size(height, width, 3, 1, 2, 3)
        lin_size = int(new_h * new_w * 128)
        
        self.lin1 = nn.Linear(lin_size, 2048)

        self.mu_lin = nn.Linear(2048, latent_size)
        self.log_var_lin = nn.Linear(2048, latent_size)

    def forward(self, x):
        x_reshaped = x.view(-1, self.height, self.width, self.num_channels).permute(0, 3, 1, 2)
        
        z = F.relu(self.conv1(x_reshaped))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = torch.flatten(z, start_dim=1)
        z = F.relu(self.lin1(z))

        mu = self.mu_lin(z)
        log_var = self.log_var_lin(z)

        return mu, log_var


class Decoder(nn.Module):

    def __init__(self, latent_size, height = 68, width = 68, num_channels = 3):
        super(Decoder, self).__init__()
        
        self.lin1 = nn.Linear(latent_size, 2048)
        self.lin2 = nn.Linear(2048, 4096)
        self.lin3 = nn.Linear(4096, 8192)
        self.lin4 = nn.Linear(8192, 16384)
        self.mu_lin = nn.Linear(16384, height * width * num_channels)
        self.kappa_lin = nn.Linear(16384, height * width * num_channels)

    def forward(self, z):
        y = F.relu(self.lin1(z))
        y = F.relu(self.lin2(y))
        y = F.relu(self.lin3(y))
        y = F.relu(self.lin4(y))
        
        x_hat = torch.sigmoid(self.mu_lin(y))
        log_kappa = self.kappa_lin(y)
                
        return x_hat, log_kappa


class VAE(nn.Module):
    def __init__(self, latent_size, height=68, width = 68, num_channels = 3):
        super(VAE, self).__init__()

        self.latent_size = latent_size
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.stats = None

        self.encoder = Encoder(latent_size, height, width, num_channels = num_channels)
        self.decoder = Decoder(latent_size, height, width, num_channels = num_channels)

    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat, log_kappa = self.decoder(sampled_z)

        return mu, log_var, x_hat, log_kappa

    def reconstruction_loss(self, x_hat, x_recon, log_kappa):
        kappa = log_kappa.exp() + 2
        
        eps = 1e-6
        x_recon = torch.clamp(x_recon, eps, 1-eps)
        x_hat = torch.clamp(x_hat, eps, 1 - eps)
        
        alfa = x_hat * (kappa - 2) + 1
        beta = (1 - x_hat) * (kappa - 2) + 1

        ln_B = lgamma(alfa) + lgamma(beta) - \
            lgamma(kappa)

        Re = - ((alfa - 1) * log(x_recon) +
               (beta - 1) * log(1 - x_recon) -
               ln_B).sum(1).mean()
            
        return Re
    
    def kl_divergence(self, log_var, mu):
        kl = 0.5 * (log_var.exp() + mu ** 2 - 1 - log_var).sum(1).mean()
        return kl
    
    def ELBO(self, x_recon, kl_beta = 1):
        mu, log_var, x_hat, log_kappa = self(x_recon)
        re = self.reconstruction_loss(x_hat, x_recon, log_kappa)
        kl = self.kl_divergence(log_var, mu)
        loss = re + kl_beta * kl
        return  loss, re, kl

    def train(self, num_epochs, dataloader, kl_beta=1, verbose=2, distribution="beta", kappa=3.47, sigma=0.229, callback=None, callback_args=dict()):
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
                
                loss, re, kl = self.ELBO(x_recon, kl_beta = kl_beta)
                
                loss.backward()
                optimizer.step()

                curr_stats[i, :] = [loss.item(), re.item(), kl.item()]

            epoch_stats = np.mean(curr_stats, axis=0)
            self.stats[epoch, :] = epoch_stats

            if verbose:
                print(f"Beta = {kl_beta}     epoch = {epoch + 1}")
                print(*epoch_stats, "\n")

            if callback is not None:
                callback(**callback_args)

        return self.stats

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("Model saved!")

    def load_model(self, filename, device="cpu"):
        map_location = device  
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print("Model loaded!")

    def encode(self, x):
        mu, _ = self.encoder(x)

        return mu.detach()

    def decode(self, z):
        x_hat, _ = self.decoder(z)

        return x_hat.detach()

class ClassifyNN(nn.Module):

    def __init__(self, num_features):
        super(ClassifyNN, self).__init__()
        
        self.lin1 = nn.Linear(num_features, 2048)
        self.lin2 = nn.Linear(2048, 512)
        self.lin3 = nn.Linear(512, 12)

    def forward(self, x):
        y = F.relu(self.lin1(x))
        y = F.relu(self.lin2(y))
        y = F.softmax(self.lin3(y), 1)
        return y
    
    def train(self, num_epochs, dataloader, eval_data=None, verbose=2):
        self.stats = np.zeros(num_epochs, 2)
        optimizer = torch.optim.Adam(self.parameters())
        
        for epoch in range(num_epochs):
            curr_loss = np.zeros(len(dataloader))
            
            for i, (X, y) in enumerate(dataloader):                
                preds = self(X)
                y = F.one_hot(y.long(), 12).float()
                loss = F.binary_cross_entropy(preds, y)
                curr_loss[i] = loss.item()

                loss.backward()
                optimizer.step()
                
                if verbose == 2:
                    print(f"Batch {i + 1} out of {len(dataloader)}")
            
            epoch_loss = np.mean(curr_loss)
            
            if eval_data is not None:
                eval_loss = self.evaluate(eval_data)
            else:
                eval_loss = 0
                
            self.stats[epoch, :] = [epoch_loss, eval_loss]
            
            if verbose:
                print(f"Epoch: {epoch + 1} / {num_epochs}")
                print(f"loss: {epoch_loss} \n")
                plt.plot(self.stats)
                plt.show()
        
        return self.stats

    def test(self, dataloader):
        for X, y in dataloader:
            preds = self(X)
            preds = torch.argmax(preds, dim=1)
            acc = torch.eq(y, preds).double().mean().item()
        
        return acc
                
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("Model saved!")