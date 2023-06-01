import matplotlib.pyplot as plt
import numpy as np
import torch


class Graphics:
    def __init__(self, model, data, num_channels=3, cmap="viridis"):
        self.model = model
        self.data = data
        self.latent_size = model.latent_size
        self.num_channels = num_channels
        self.cmap=cmap

    def random_generations(self, filename=""):
        fig, axs = plt.subplots(3, 3)
        for ax in axs.flatten():
            rand_z = torch.randn((1, self.model.latent_size))
            generation = self.model.decode(rand_z)
            ax.imshow(generation.reshape(68, 68, self.num_channels), cmap=self.cmap)
            ax.set_xticks([])
            ax.set_yticks([])

        if filename:
            plt.savefig(filename, dpi=300)

        plt.show()

    def reconstructions(self, indxs=[0, 1, 2, 3], filename=""):
        fig, axs = plt.subplots(len(indxs), 2)
        for i, ax in enumerate(axs):
            cell = self.data[i]
            z = self.model.encode(cell)
            x_hat = self.model.decode(z)
            ax[0].imshow(cell.reshape(68, 68, self.num_channels), cmap=self.cmap )
            ax[1].imshow(x_hat.reshape(68, 68, self.num_channels), cmap=self.cmap)

        fig.set_size_inches(4, len(indxs) * 2)

        if filename:
            plt.savefig(filename, dpi=300)

        plt.show()

    def show_convergence(self, filename=""):
        fig, axs = plt.subplots(1, 3)
        titles = ["Loss", "Re", "kl"]
        mask = np.sum(self.model.stats, axis=1) != 0
        for i, ax in enumerate(axs.flatten()):
            ax.plot(self.model.stats[mask, i])
            ax.set_title(titles[i])

        if filename:
            plt.savefig(filename, dpi=300)

        plt.show()

    def investigate_feature(self, feature_idx=0, data_idx=0, filename=""):
        fig, axs = plt.subplots(5, 5)
        cell = self.data[data_idx]
        mu = self.model.encode(cell)
        space = np.linspace(-2, 2, 25)
        for i, ax in zip(space, axs.flatten()):
            mu[0, feature_idx] = i
            decoding = self.model.decode(mu)
            ax.imshow(decoding.reshape(68, 68, self.num_channels), cmap=self.cmap)
            ax.set_xticks([])
            ax.set_yticks([])

        if filename:
            plt.savefig(filename, dpi=300)

        plt.show()

    def interpolate(self, idxs=[0, 1], filename=""):
        # interpolerer mellem to billeder
        cell1 = self.data[idxs[0]]
        cell2 = self.data[idxs[1]]
        encoding1 = self.model.encode(cell1)
        encoding2 = self.model.encode(cell2)

        retning = encoding2 - encoding1

        fig, axs = plt.subplots(4, 4)
        for i, ax in enumerate(axs.flatten()):
            step = i / 16 * retning
            new_z = encoding1 + step
            generated_face = self.model.decode(new_z)
            ax.imshow(generated_face.reshape(68, 68, self.num_channels), cmap=self.cmap)
            ax.set_xticks([])
            ax.set_yticks([])

        if filename:
            plt.savefig(filename, dpi=300)

        plt.show()

    def plot_encoding(self, idx=0, filename=""):
        eps = 1e-6
        x_hat = self.data[idx]
        z = self.model.encode(x_hat)
        plt.bar(np.arange(self.latent_size), z.flatten())
        mask = (-eps < z.flatten()) & (z.flatten() < eps).int()
        alpha_hat = np.mean(mask.numpy())
        plt.title(rf"$\alpha = {alpha_hat}$")

        if filename:
            plt.savefig(filename, dpi=300)

        plt.show()
