import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from torch.utils.data import Dataset
import torch.nn as nn

# Definerer variable
input_dim = 62 * 47 * 3
batch_size = 128
num_epochs = 100
learning_rate = 0.001
latent_size = 300
beta = 0.1
delta_beta  = 0.002   ## ændringen i beta-værdi for hver epoch
eps = 1e-6
alpha = 0.3

class Faces(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self):
        faces, _ = fetch_lfw_people(return_X_y=True, color=True)

        self.data = faces

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


train_data = Faces()


class Encoder(torch.nn.Module):
    def __init__(self, network):
        super(Encoder, self).__init__()
        self.network = network
        
    def forward(self, x):
        x_reshaped = x.view(-1, 62, 47, 3).permute(0, 3, 1, 2)
        z = self.network(x_reshaped)
        mu, log_var, log_spike = torch.chunk(z, 3, 1)
        return mu, log_var, log_spike
    
class Decoder(torch.nn.Module):
    def __init__(self, network):
        super(Decoder, self).__init__()
        self.network = network

    def forward(self, z):
        x_hat = self.network(z)
        return x_hat


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def reparameterization(self, mu, log_var, log_spike):
        std = torch.exp(0.5 * log_var)
        eps_ = torch.randn_like(std)
        gaussian = eps_.mul(std).add_(mu)
        eta = torch.randn_like(std)
        
        c = 50
        selection = torch.sigmoid(c * (eta + log_spike.exp() - 1))
        return selection.mul(gaussian)

    def forward(self, x):
        mu, log_var, log_spike = self.encoder(x)
        z = self.reparameterization(mu, log_var, log_spike)
        x_hat = self.decoder(z)
        
        return mu, log_var, log_spike, z, x_hat

    def encode(self, x):
        mu, log_var, log_spike = self.encoder(x)
        
        ## samplet et z
        # z = self.reparameterization(mu, log_var, log_spike)
        
        ## måske en mere rigtig måde at gøre det
        spike = log_spike.exp()
        selection = torch.sigmoid(50 * (spike - 1))
        z = selection.mul(mu)
        
        return z.detach().numpy()

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat.detach().numpy()


encoder_network = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
    nn.MaxPool2d((2, 2)),
    nn.LeakyReLU(),
    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
    nn.MaxPool2d((2, 2)),
    nn.LeakyReLU(),
    nn.Flatten(),
    nn.Linear(64*15*11, 3*latent_size)
)

decoder_network = nn.Sequential(
    nn.Linear(latent_size, 2048),
    nn.ReLU(),
    nn.Linear(2048, 8192),
    nn.ReLU(),
    nn.Linear(8192, input_dim),
    nn.Sigmoid()
)

dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size,
    shuffle=True)


encoder = Encoder(encoder_network)
decoder = Decoder(decoder_network)

vae = VAE(encoder, decoder)

# %%

stats = np.zeros((num_epochs, 4))
random_z = torch.randn((1, latent_size))
random_face = train_data[np.random.randint(len(train_data))]

optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for data in dataloader:
        inputs = data
        inputs = inputs.view(-1, input_dim)
        optimizer.zero_grad()
        mu, log_var, log_spike, z, x_hat = vae(inputs)
        
        Re = F.binary_cross_entropy(x_hat, inputs, reduction="sum")
        
        spike = torch.clamp(log_spike.exp(), eps, 1.0 - eps) 
        alpha = 0.3

        prior1 = -0.5 * torch.sum(spike.mul(1 + log_var - mu.pow(2) - log_var.exp()))
        prior21 = (1 - spike).mul(torch.log((1 - spike) / (1 - alpha)))
        prior22 = spike.mul(torch.log(spike / alpha))
        prior2 = torch.sum(prior21 + prior22)
        PRIOR = prior1 + prior2

        loss = Re + beta * PRIOR
        loss.backward()
        optimizer.step()

    beta += delta_beta
    
    curr_stats = [epoch, loss.item(), Re.item(), PRIOR.item()]
    stats[epoch+1, :] = curr_stats
    print(*curr_stats)
            
    z_ = z[0].detach().numpy()
    plt.bar(np.arange(len(z_)), z_)
    plt.show()

    fig, ax = plt.subplots(1, 3)
    generation = vae.decode(random_z)
    ax[0].imshow(generation.reshape(62, 47, 3))
    encoding = vae.encode(torch.tensor(random_face))
    decoding = vae.decode(torch.tensor(encoding))
    ax[2].imshow(decoding.reshape(62, 47, 3))
    ax[1].imshow(random_face.reshape(62, 47, 3))
    plt.show()

    fig, axs = plt.subplots(1, 3)
    for i, ax in enumerate(axs.flatten()):
        ax.plot(stats[:epoch, i+1])
    plt.show()

np.save("betabce", stats)
torch.save(vae.state_dict(), "betabce")

# %%

## Eventuelt hent model her

encoder = Encoder(encoder_network)
decoder = Decoder(decoder_network)
vae = VAE(encoder, decoder)
vae.load_state_dict(torch.load("betabce"))


# %%

random_zs = torch.randn((10, latent_size))
for z in random_zs:
    x_hat = vae.decode(z)
    plt.imshow(x_hat.reshape(62, 47, 3))
    plt.show()

# %%

random_faces = train_data[np.random.randint(len(train_data), size=10)]
for face in random_faces:
    encoding = vae.encode(torch.tensor(face))
    decoding = vae.decode(torch.tensor(encoding))
    plt.imshow(face.reshape(62, 47, 3))
    plt.show()
    plt.imshow(decoding.reshape(62, 47, 3))
    plt.show()

# %%
idx = np.random.randint(len(train_data))
random_face = train_data[idx]
encoding = vae.encode(torch.tensor(random_face))
encoding_idx = np.where((-eps > encoding) & (encoding < eps))[1][0]
space = np.linspace(-2, 2, 20)
for i in space:
    encoding[0, encoding_idx] = i
    decoding = vae.decode(torch.tensor(encoding))
    plt.imshow(decoding.reshape(62, 47, 3))
    plt.show()
