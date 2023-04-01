#%%
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
from torch.distributions import kl_divergence, Normal
from torch import log, lgamma
from torchsummary import summary

#%%
# Definerer variable

batch_size = 128
num_epochs = 10
learning_rate = 0.001
latent_size = 800
eps = 1e-6
name_of_model = "beta_bce"

### Loading data and standardizing ###
class Cells(Dataset):
    """BBBC021 dataset."""

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

#%%

train_data = Cells()


def cyclical(epoch, interval, min_, max_):
    x = epoch % interval
    a = (max_ - min_) / (interval / 2)
    return min(a * x + min_, max_)


class Encoder(torch.nn.Module):
    def __init__(self, network):
        super(Encoder, self).__init__()
        self.network = network

    def forward(self, x):
        x_reshaped = x.view(-1, 68, 68, 3).permute(0, 3, 1, 2)
        z = self.network(x_reshaped)
        mu, log_var = torch.chunk(z, 2, 1)
        return mu, log_var


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

    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps_ = torch.randn_like(std)
        new_z = mu + eps_ * std

        return new_z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decoder(sampled_z)

        return mu, log_var, sampled_z, x_hat

    def encode_as_np(self, x):
        mu, log_var = self.encoder(x)
        return mu.detach().numpy()

    def decode_as_np(self, z):
        x_hat = self.decoder(z)
        return x_hat.detach().numpy()


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

decoder_network = nn.Sequential(
    nn.Linear(latent_size, 1024),
    nn.LeakyReLU(),
    
    nn.Linear(1024, 4096),
    nn.LeakyReLU(),

    nn.Linear(4096, 68*68*3),
    nn.Sigmoid()
)

print("Encoder network:")
summary(encoder_network, (3, 68, 68))
print("\n" * 3)
print("Decoder network:")
summary(decoder_network, (1, latent_size))


dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size,
    drop_last=True, shuffle=True)


encoder = Encoder(encoder_network)
decoder = Decoder(decoder_network)

vae = VAE(encoder, decoder)

stats = np.zeros((num_epochs, 3))

# %%
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

## for debugging
torch.set_anomaly_enabled(True)


### Training Loop
for epoch in range(num_epochs):
    kl_beta = 70

    dataloader_iterations = int(len(train_data) / batch_size)
    epoch_stats = np.zeros((dataloader_iterations, 3))

    for i, x in enumerate(dataloader):
        #print(f'batch number {i} out of {dataloader_iterations}')
        optimizer.zero_grad()
        mu, log_var, sampled_z, x_hat = vae(x)
        std = torch.exp(0.5 * log_var)

        x = torch.clamp(x, eps, 1-eps)
        x_hat = torch.clamp(x_hat, eps, 1 - eps)

        precision = 100
        alfa = precision * x_hat
        beta = precision * (1 - x_hat)

        ln_B = lgamma(alfa) + lgamma(beta) - lgamma(torch.tensor(precision))
        Re = -((alfa - 1) * log(x) + (beta - 1)
               * log(1 - x) - ln_B).sum(1).mean()

        kl = kl_divergence(
            Normal(mu, std),
            Normal(0, 1)
        ).sum(1).mean()

        loss = Re + kl_beta * kl
        loss.backward()
        optimizer.step()

        epoch_stats[i, :] = [loss.item(), Re.item(), kl.item()]

    curr_stats = np.mean(epoch_stats, axis=0)
    stats[epoch, :] = curr_stats

    print("Beta = ", kl_beta)
    print(*curr_stats, "\n")

    fig, axs = plt.subplots(3, 3)
    for ax in axs.flatten():
        rand_z = torch.randn((1, latent_size))
        generation = vae.decode_as_np(rand_z)
        ax.imshow(generation.reshape(68, 68, 3))
    plt.show()

    fig, axs = plt.subplots(4, 2)
    for ax in axs:
        rand_indx = np.random.randint(len(train_data))
        random_face = train_data[rand_indx]
        z, _ = vae.encoder(random_face)
        x_hat = vae.decode_as_np(z)
        ax[0].imshow(random_face.reshape(68, 68, 3))
        ax[1].imshow(x_hat.reshape(68, 68, 3))
    fig.set_size_inches(4, 10)
    plt.show()

    fig, axs = plt.subplots(1, 3)
    titles = ["Loss", "Re", "kl"]
    for i, ax in enumerate(axs.flatten()):
        ax.plot(stats[:epoch+1, i])
        ax.set_title(titles[i])
    plt.show()

np.save(name_of_model, stats)
torch.save(vae.state_dict(), name_of_model)
# %%

"""Eventuelt hent model her"""
encoder = Encoder(encoder_network)
decoder = Decoder(decoder_network)
vae = VAE(encoder, decoder)
vae.load_state_dict(torch.load("beta_bce"))

#%%

## laver tilfældigt genererede billeder
random_zs = torch.randn((10, latent_size))
for z in random_zs:
    x_hat = vae.decode_as_np(z)
    plt.imshow(x_hat.reshape(68, 68, 3))
    plt.show()
# %%

## laver en rekonstruktion af et tilfældigt ansigt
random_faces = train_data[np.random.randint(len(train_data), size=10)]
for face in random_faces:
    mu, _ = vae.encoder(torch.tensor(face))
    decoding = vae.decode_as_np(torch.tensor(mu))
    plt.imshow(face.reshape(68, 68, 3))
    plt.show()
    plt.imshow(decoding.reshape(68, 68, 3))
    plt.show()
# %%

## ændrer på en latent variabel over tid
fig, axs = plt.subplots(5,5)
idx = np.random.randint(len(train_data))
random_face = train_data[idx]
mu = vae.encode_as_np(random_face)
space = np.linspace(-2, 2, 25)
encoding_idx = 0
for i, ax in zip(space, axs.flatten()):
    mu[0, 2] = i
    decoding = vae.decode_as_np(torch.tensor(mu))
    ax.imshow(decoding.reshape(68, 68, 3))
    
#%%

## interpolerer mellem to billeder
randomface1 = train_data[np.random.randint(len(train_data))]
randomface2 = train_data[np.random.randint(len(train_data))]
encoding1, _ = vae.encoder(randomface1)
encoding2, _ = vae.encoder(randomface2)

fig, axs = plt.subplots(1,2)
axs[0].imshow(randomface1.view(68,68,3))
axs[1].imshow(randomface2.view(68,68,3))

retning = encoding2 - encoding1

fig, axs = plt.subplots(4,4)
for i, ax in enumerate(axs.flatten()):
    step = i / 16 * retning
    new_z = encoding1 + step
    generated_face = vae.decode_as_np(new_z)
    ax.imshow(generated_face.reshape(68,68,3))
plt.savefig("interpolations", dpi=300)