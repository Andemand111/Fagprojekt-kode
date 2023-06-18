import torch
import numpy as np
from torch import log, lgamma
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(torch.nn.Module):
    """
    Encoder class for a variational autoencoder.

    Args:
        latent_size (int): The size of the latent space.
        num_channels (int): The number of input channels (default: 3).
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
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            mu (torch.Tensor): Mean of the latent space.
            log_var (torch.Tensor): Log variance of the latent space.
        """
        
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
    """
    Decoder class for a variational autoencoder.

    Args:
        latent_size (int): The size of the latent space.
        num_channels (int): The number of output channels (default: 3).
    """

    def __init__(self, latent_size, num_channels = 3):
        super(Decoder, self).__init__()

        self.lin1 = nn.Linear(latent_size, 2048)
        self.lin2 = nn.Linear(2048, 4096)
        self.lin3 = nn.Linear(4096, 68 * 68 * num_channels)

    def forward(self, z):
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Latent space tensor.

        Returns:
            x_hat (torch.Tensor): Reconstructed output tensor.
        """
        
        x_hat = F.relu(self.lin1(z))
        x_hat = F.relu(self.lin2(x_hat))
        x_hat = torch.sigmoid(self.lin3(x_hat))
        return x_hat


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    Args:
        latent_size (int): The size of the latent space.
        num_channels (int): The number of input/output channels (default: 3).
    """
    
    def __init__(self, latent_size, num_channels = 3):
        super(VAE, self).__init__()

        self.latent_size = latent_size
        self.num_channels = num_channels
        self.stats = None

        self.encoder = Encoder(latent_size, num_channels = num_channels)
        self.decoder = Decoder(latent_size, num_channels = num_channels)

    def reparameterization(self, mu, log_var):
        """
        Reparameterization trick for sampling from a normal distribution.

        Args:
            mu (torch.Tensor): Mean of the latent space.
            log_var (torch.Tensor): Log variance of the latent space.

        Returns:
            z (torch.Tensor): Sampled latent space tensor.
        """
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            mu (torch.Tensor): Mean of the latent space.
            log_var (torch.Tensor): Log variance of the latent space.
            x_hat (torch.Tensor): Reconstructed output tensor.
        """

        mu, log_var = self.encoder(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decoder(sampled_z)

        return mu, log_var, x_hat

    def reconstruction_loss(self, x_hat, x_recon, distribution="beta", kappa=3.47, sigma=0.229):
        """
        Compute the reconstruction loss between the reconstructed output and the original input.

        Args:
            x_hat (torch.Tensor): Reconstructed output tensor.
            x_recon (torch.Tensor): Original input tensor.
            distribution (str): Distribution type ("beta" or "normal").
            kappa (float): Parameter for the beta distribution (default: 3.47).
            sigma (float): Standard deviation for the normal distribution (default: 0.229).

        Returns:
            Re (torch.Tensor): Reconstruction loss.
            
        """
        
        if distribution == "beta":
            eps = 1e-6
            x_recon = torch.clamp(x_recon, eps, 1-eps)
            x_hat = torch.clamp(x_hat, eps, 1 - eps)
            
            alfa = x_hat * (kappa - 2) + 1
            beta = (1 - x_hat) * (kappa - 2) + 1

            ln_B = lgamma(alfa) + lgamma(beta) - \
                lgamma(torch.tensor(kappa))

            Re = - ((alfa - 1) * log(x_recon) +
                   (beta - 1) * log(1 - x_recon) -
                   ln_B).sum(1).mean()

        elif distribution == "normal":
            var = sigma ** 2
            distance = ((x_hat - x_recon) ** 2).sum(1).mean() / (2 * var)
            const = 0.5 * 68**2 * self.num_channels * log(torch.tensor([2 * torch.pi * var])) 
            Re = distance.to(device) + const.to(device)
            
        return Re
    
    def kl_divergence(self, log_var, mu):
        """
        Compute the Kullback-Leibler divergence between the learned distribution and a standard normal distribution.

        Args:
            log_var (torch.Tensor): Log variance of the latent space.
            mu (torch.Tensor): Mean of the latent space.

        Returns:
            kl (torch.Tensor): KL divergence.
        """
        
        kl = 0.5 * (log_var.exp() + mu ** 2 - 1 - log_var).sum(1).mean()
        return kl
    
    def ELBO(self, x_recon, distribution = "beta", kappa = 3.47, sigma = 0.229, kl_beta = 1):
        """
        Compute the Evidence Lower Bound (ELBO) loss.

        Args:
            x_recon (torch.Tensor): Original input tensor.
            distribution (str): Distribution type ("beta" or "normal").
            kappa (float): Parameter for the beta distribution (default: 3.47).
            sigma (float): Standard deviation for the normal distribution (default: 0.229).
            kl_beta (float): Weight for the KL divergence term (default: 1).

        Returns:
            loss (torch.Tensor): ELBO loss.
            re (torch.Tensor): Reconstruction loss.
            kl (torch.Tensor): KL divergence.
        """
        
        mu, log_var, x_hat = self(x_recon)
        re = self.reconstruction_loss(x_hat, x_recon, distribution, kappa, sigma)
        kl = self.kl_divergence(log_var, mu)
        loss = re + kl_beta * kl
        return  loss, re, kl

    def train(self, num_epochs, dataloader, kl_beta=1, verbose=2, distribution="beta", kappa=3.47, sigma=0.229, callback=None, callback_args=dict()):
        """
        Training loop for the VAE.

        Args:
            num_epochs (int): Number of training epochs.
            dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
            kl_beta (float): Weight for the KL divergence term (default: 1).
            verbose (int): Verbosity level (0: silent, 1: print epoch statistics, 2: print epoch and batch statistics).
            distribution (str): Distribution type ("beta" or "normal").
            kappa (float): Parameter for the beta distribution (default: 3.47).
            sigma (float): Standard deviation for the normal distribution (default: 0.229).
            callback (function): Callback function to be called at the end of each epoch (default: None).
            callback_args (dict): Arguments to be passed to the callback function (default: {}).

        Returns:
            stats (np.ndarray): Array of training statistics (loss, reconstruction loss, KL divergence) for each epoch.
        """
        
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
                
                loss, re, kl = self.ELBO(x_recon, distribution=distribution, kappa=kappa, sigma=sigma, kl_beta = kl_beta)
                
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
        mu, log_var = self.encoder(x)

        return mu.detach()

    def decode(self, z):
        x_hat = self.decoder(z)

        return x_hat.detach()

class ClassifyNN(nn.Module):
    
    """
    Neural network model for classification tasks.

    This class defines a neural network model with customizable activation function and hidden layers.
    It provides methods for training, testing, saving and loading the model.

    Args:
        num_features (int): Number of input features.
        num_hidden (int): Number of units in the hidden layer.
        activation (str): Activation function to use (default: "relu").
    """

    def __init__(self, num_features, num_hidden, activation="relu"):
        super(ClassifyNN, self).__init__()
        self.activation = activation
        
        if self.activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        
        self.lin1 = nn.Linear(num_features, num_hidden)
        self.lin2 = nn.Linear(num_hidden, num_hidden)
        self.lin3 = nn.Linear(num_hidden, num_hidden)
        self.lin4 = nn.Linear(num_hidden, 12)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        y = self.act(self.lin1(x))
        y = self.act(self.lin2(y))
        y = self.act(self.lin3(y))
        y = F.softmax(self.lin4(y), 1)
        return y
    
    def train(self, num_epochs, dataloader, eval_data=None, verbose=2):
        """
        Trains the model.

        Args:
            num_epochs (int): Number of training epochs.
            dataloader (torch.utils.data.DataLoader): Training data loader.
            eval_data (torch.utils.data.DataLoader): Evaluation data loader (default: None).
            verbose (int): Verbosity level (0: silent, 1: print epoch summary, 2: print batch details) (default: 2).

        Returns:
            numpy.ndarray: Training statistics.
        """
        
        self.stats = np.zeros((num_epochs, 2))
        optimizer = torch.optim.Adam(self.parameters())
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            curr_loss = np.zeros(len(dataloader))
            
            for i, (X, y) in enumerate(dataloader):      
                optimizer.zero_grad()
                
                preds = self(X)
                loss = criterion(preds, y.long())
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
                # plt.plot(self.stats)
                # plt.show()
        
        return self.stats

    def test(self, dataloader):
        """
       Evaluates the model on test data.

       Args:
           dataloader (torch.utils.data.DataLoader): Test data loader.

       Returns:
           float: Accuracy of the model on the test data.
       """
        
        accs = np.zeros(len(dataloader))
        
        for X, y in dataloader:
            preds = self(X)
            preds = torch.argmax(preds, dim=1)
            curr_acc = torch.eq(y, preds).double().mean().item()
            accs = np.append(accs, curr_acc)
        
        final_acc = np.mean(accs)
        
        return final_acc
                
    def save_model(self, filename):
        """
        Saves the model's state dictionary to a file.

        Args:
            filename (str): Name of the file to save the model.
        """
        
        torch.save(self.state_dict(), filename)
        print("Model saved!")
        
    def load_model(self, filename, device="cpu"):
        """
        Loads a trained model from a file.

        Args:
            filename (str): Name of the file to load the model from.
            device (str): Device to load the model onto (default: "cpu").
        """
        
        map_location = device  
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print("Model loaded!")
