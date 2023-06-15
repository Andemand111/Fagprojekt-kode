from models import VAE
from dataset import SmallCells
from graphics import Graphics
import matplotlib.pyplot as plt

import numpy as np
import torch
#%%

data = np.load("G:/Mit drev/Uni/4. semester/fagprojekt/fra_git/celle_data.npy")
dataset = SmallCells(data)

#%%
path = "G:/Mit drev/Project Splinter Cell/modeller/rgbmodeller/"
name = "rgb_model_beta64"
latent_size = 64
num_channels = 3
channel = None
cmap = "viridis"
title= "Beta-VAE"

model = VAE(latent_size, num_channels=num_channels)
model.load_model(path + name)
model.height = model.width = 68
plots = Graphics(model, dataset, channel=channel, cmap=cmap)

filename = f"_{name}"
plots.random_generations(title="Random generations, " + title, filename="random" + filename)
plots.reconstructions(title="Reconstructions, " + title, filename="reconstruction" + filename)
plots.interpolate(title="Interpolations, " + title, filename="interpolate" + filename)

z = plots.plot_encoding(title=title, filename="encoding" + filename)
interesting_feature = np.argmax(abs(z))

plots.investigate_feature(interesting_feature)

#%%
n_rows = 10
sim_scores = plots.find_interesting_features()
feature_i, feature_j = np.argsort(sim_scores)[:2]
space = torch.linspace(-2,2,n_rows)
fig, axs = plt.subplots(n_rows, n_rows)
rand_z = torch.rand((1, latent_size))
for i in range(n_rows):
    for j in range(n_rows):
        rand_z_copy = torch.clone(rand_z)
        rand_z_copy[0, feature_i] = space[i]
        rand_z_copy[0, feature_j] = space[j]
        decoding = model.decode(rand_z_copy).reshape(68,68,num_channels)
        axs[i, j].imshow(decoding, cmap=cmap)
        
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])

plt.suptitle(f"Varying two features, {title}")
plt.savefig(f"varying_features_{name}", dpi=300, bbox_inches="tight")

