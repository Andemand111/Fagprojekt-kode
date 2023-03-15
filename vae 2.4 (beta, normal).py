import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from torch.utils.data import Dataset
from torch import nn

input_dim = 62 * 47 * 3
batch_size = 128
num_epochs = 100
learning_rate = 0.001
latent_size = 800
beta = 1

#%%

class Faces(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self, mode='train'):
        faces, _ = fetch_lfw_people(return_X_y=True, color=True)
        faces = faces * 2 - 1
        
        data_size = len(faces)
        idx1 = int(0.7 * data_size)
        idx2 = int(0.85 * data_size)

        if mode == 'train':
            self.data = faces[:idx1]
        elif mode == 'val':
            self.data = faces[idx1:idx2]
        else:
            self.data = faces[idx2:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
    
    def showim(self, ax, idx):
        rescaled = (self.data[idx] + 1) / 2
        ax.imshow(rescaled.reshape(62, 47, 3))
    
train_data = Faces("train")

def showim(vec,  ax):
    ax.imshow(vec.reshape(62,47,3))
    
fig, axs = plt.subplots(3,3,subplot_kw={'xticks': [], 'yticks': []})
for ax in axs.flatten():
    train_data.showim(ax, np.random.randint(len(train_data)))

#%%

class Encoder(torch.nn.Module):
    def __init__(self, network):
        super(Encoder, self).__init__()
        self.network = network

    def forward(self, x):
        x_reshaped = x.view(-1,62,47,3).permute(0,3,1,2)
        z = self.network(x_reshaped)
        mu, log_std = torch.chunk(z, 2, 1)
        std = torch.exp(log_std)

        return torch.distributions.Normal(loc=mu, scale=std)


class Decoder(torch.nn.Module):
    def __init__(self, network):
        super(Decoder, self).__init__()
        self.network = network
        
    def forward(self, z):
        mu = self.network(z)

        return torch.distributions.Normal(mu, torch.ones_like(mu))

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, state):
        q_z = self.encoder(state)
        # Parameterization trick
        z = q_z.rsample()
        return self.decoder(z), q_z
    
    def encode(self, x):
        z_distribution = self.encoder(x)
        z = z_distribution.loc
        return z.detach().numpy()
    
    def decode(self, z):
        decoding_dist = self.decoder(z)
        x_hat = decoding_dist.loc
        return x_hat.detach().numpy()
    
    def decode_as_im(self, z):
        decoding = self.decode(z)
        return (decoding + 1) / 2

dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size,
    shuffle=True)

print('Number of samples: ', len(train_data))

encoder_network = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=(3,3), padding=(1,1)),
    nn.MaxPool2d((2,2)),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1)),
    nn.MaxPool2d((2,2)),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64*15*11, 2*latent_size)
)

decoder_network = nn.Sequential(
    nn.Linear(latent_size, 2048),
    nn.ReLU(),
    nn.Linear(2048, 4096),
    nn.ReLU(),
    nn.Linear(4096, input_dim),
    nn.Tanh()
)

encoder = Encoder(encoder_network)
decoder = Decoder(decoder_network)

vae = VAE(encoder, decoder)

stats = np.zeros((num_epochs, 4))

optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for data in dataloader:
        inputs = data
        
        optimizer.zero_grad()
        p_x, q_z = vae(inputs)
        
        log_likelihood = p_x.log_prob(inputs).sum(-1).mean()
        kl = torch.distributions.kl_divergence(
            q_z, 
            torch.distributions.Normal(0, 1.)
        ).sum(-1).mean()
        
        loss = -(log_likelihood - beta * kl)
        loss.backward()
        optimizer.step()
        l = loss.item()
    
    curr_stats = [epoch, l, log_likelihood.item(), kl.item() * beta]
    stats[epoch, :] = curr_stats
    print(*curr_stats)
    
#%%

names = [r"Loss", "Log_likelihood", "$\\beta$ KL"]
fig, axs = plt.subplots(3)
fig.set_size_inches(8, 13)

for i, ax in enumerate(axs.flatten()):
    ax.set_title(names[i])
    ax.plot(stats[:, 1+i])

#%%


fig, axs = plt.subplots(3,3,subplot_kw={'xticks': [], 'yticks': []})
for ax in axs.flatten():
    random_z = 4 * torch.randn((1,latent_size))
    x_hat = (vae.decode(random_z) + 1)/2
    ax.imshow(x_hat.reshape(62,47,3))
    
plt.savefig("results from normal")
    
    #%%

# Plot function:
# Side-by-Side, original vs. reconstructed
for i in range(5):
    fig, axs = plt.subplots(2)
    idx = np.random.randint(len(train_data))
    pic = train_data[idx].reshape(1,-1)
    z = vae.encode(torch.tensor(pic))
    generation = vae.decoder(torch.tensor(z)).loc.detach().numpy()
    axs[0].imshow(((pic + 1)/2).reshape(62,47,3))
    axs[1].imshow(((generation + 1)/2).reshape(62,47,3))
    
    
    
    
