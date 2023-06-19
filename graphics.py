import matplotlib.pyplot as plt  
import numpy as np  
import torch  
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class Graphics:
    """
    Class for visualizing generations coming from a VAE.

    Args:
        model (object): The model used for generating and reconstructing images.
        data (numpy.ndarray): The input data used for reconstruction.
        channel (int): The channel index to visualize (optional).
        cmap (str): The color map to use for visualization (default: "viridis").
    """
    
    def __init__(self, model, data, channel=None, cmap="viridis"):
        """
        Initializes the Graphics class.

        Args:
            model (object): The model used for generating and reconstructing images.
            data (numpy.ndarray): The input data used for reconstruction.
            channel (int): The channel index to visualize (optional).
            cmap (str): The color map to use for visualization (default: "viridis").
        """
        self.model = model  # Assigning the model
        self.data = data  # Assigning the input data
        self.latent_size = model.latent_size  # Assigning the latent size from the model
        self.num_channels = model.num_channels  # Assigning the number of channels from the model
        self.cmap = cmap  # Assigning the color map
        self.channel = channel  # Assigning the channel index

    def random_generations(self, title = "", filename=""):
        """
        Generates random images using the trained model.

        Args:
            title (str): The title of the visualization (optional).
            filename (str): The filename to save the generated images (optional).
        """
        fig, axs = plt.subplots(3, 3)  # Creating a 3x3 grid of subplots
        for ax in axs.flatten():
            rand_z = torch.randn((1, self.model.latent_size))  # Generating random latent vectors
            generation = self.model.decode(rand_z)  # Decoding the latent vectors to generate images
            ax.imshow(generation.reshape(68, 68, self.num_channels), cmap=self.cmap)  # Plotting the generated images
            ax.set_xticks([])  # Removing x-axis ticks
            ax.set_yticks([])  # Removing y-axis ticks

        if filename:
            plt.savefig(filename, dpi=300)  # Saving the generated images if a filename is provided
            
        if title:
            fig.suptitle(title)  # Adding a title to the visualization

        plt.show()  # Displaying the plot

    def reconstructions(self, indxs=[0, 1, 2, 3], title="", filename=""):
        """
        Reconstructs and visualizes the input images using the trained model.

        Args:
            indxs (list): The indices of the input images to reconstruct (default: [0, 1, 2, 3]).
            title (str): The title of the visualization (optional).
            filename (str): The filename to save the reconstructed images (optional).
        """
        fig, axs = plt.subplots(len(indxs), 2)  # Creating subplots for reconstructed images
        for i, ax in enumerate(axs):
            cell = self.data[indxs[i]]  # Selecting an input image
            if self.num_channels == 1:
                cell = cell.reshape(68, 68, 3)[:, :, self.channel]
            else:
                cell = cell.reshape(68, 68, 3)
            z = self.model.encode(cell.flatten())  # Encoding the input image to obtain latent vectors
            x_hat = self.model.decode(z)  # Decoding the latent vectors to reconstruct the image
            ax[0].imshow(cell, cmap=self.cmap)  # Plotting the original image
            ax[1].imshow(x_hat.reshape(68, 68, self.num_channels), cmap=self.cmap)  # Plotting the reconstructed image

        fig.set_size_inches(4, len(indxs) * 2)  # Setting the figure size

        if filename:
            plt.savefig(filename, dpi=300)  # Saving the reconstructed images if a filename is provided
        
        if title:
            fig.suptitle(title)  # Adding a title to the visualization

        plt.show()  # Displaying the plot

    def show_convergence(self, title="", filename=""):
        """
        Plots the convergence during training (e.g., loss, reconstruction error, KL divergence).

        Args:
            title (str): The title of the visualization (optional).
            filename (str): The filename to save the convergence plot (optional).
        """
        fig, axs = plt.subplots(1, 3)  # Creating subplots for convergence plots
        titles = ["Loss", "Re", "kl"]  # Titles for each convergence plot
        mask = np.sum(self.model.stats, axis=1) != 0  # Filtering out zero rows from the stats array
        for i, ax in enumerate(axs.flatten()):
            ax.plot(self.model.stats[mask, i])  # Plotting the convergence data
            ax.set_title(titles[i])  # Setting the title for each subplot

        if filename:
            plt.savefig(filename, dpi=300)  # Saving the convergence plot if a filename is provided
            
        if title:
            fig.suptitle(title)  # Adding a title to the visualization

        plt.show()  # Displaying the plot

    def investigate_feature(self, feature_idx=0, data_idx=0, title="", filename=""):
        """
       Investigates the effect of a specific feature by varying its value.

       Args:
           feature_idx (int): The index of the feature to investigate (default: 0).
           data_idx (int): The index of the input image to use as a reference (default: 0).
           title (str): The title of the visualization (optional).
           filename (str): The filename to save the visualization (optional).
       """
        fig, axs = plt.subplots(5, 5)  # Creating subplots for feature investigation
        cell = self.data[data_idx]  # Selecting an input image as a reference
        if self.num_channels == 1:
            cell = cell.reshape(68, 68, 3)[:, :, self.channel]
        mu = self.model.encode(cell.flatten())  # Encoding the reference image to obtain latent vectors
        space = np.linspace(-2, 2, 25)  # Creating a range of values to investigate the feature
        for i, ax in zip(space, axs.flatten()):
            mu[0, feature_idx] = i  # Setting the feature value to investigate
            decoding = self.model.decode(mu)  # Decoding the modified latent vectors to generate images
            ax.imshow(decoding.reshape(68, 68, self.num_channels), cmap=self.cmap)  # Plotting the generated images
            ax.set_xticks([])  # Removing x-axis ticks
            ax.set_yticks([])  # Removing y-axis ticks

        if filename:
            plt.savefig(filename, dpi=300)  # Saving the visualization if a filename is provided
            
        if title:
            fig.suptitle(title)  # Adding a title to the visualization

        plt.show()  # Displaying the plot

    def interpolate(self, idxs=[0, 1], title="", filename=""):
        """
        Interpolates between two images and visualizes the intermediate steps.

        Args:
            idxs (list): The indices of the two input images to interpolate between (default: [0, 1]).
            title (str): The title of the visualization (optional).
            filename (str): The filename to save the interpolated images (optional).
        """
        cell1 = self.data[idxs[0]]  # Selecting the first input image
        cell2 = self.data[idxs[1]]  # Selecting the second input image
        
        if self.num_channels == 1:
            cell1 = cell1.reshape(68, 68, 3)[:, :, self.channel]
            cell2 = cell2.reshape(68, 68, 3)[:, :, self.channel]
        
        encoding1 = self.model.encode(cell1.flatten())  # Encoding the first input image to obtain latent vectors
        encoding2 = self.model.encode(cell2.flatten())  # Encoding the second input image to obtain latent vectors

        retning = encoding2 - encoding1  # Calculating the direction between the two encodings

        fig, axs = plt.subplots(4, 4)  # Creating subplots for interpolated images
        for i, ax in enumerate(axs.flatten()):
            step = i / 16 * retning  # Calculating the step size for interpolation
            new_z = encoding1 + step  # Interpolating between the two encodings
            generated_face = self.model.decode(new_z)  # Decoding the interpolated latent vectors to generate images
            ax.imshow(generated_face.reshape(68, 68, self.num_channels), cmap=self.cmap)  # Plotting the interpolated images
            ax.set_xticks([])  # Removing x-axis ticks
            ax.set_yticks([])  # Removing y-axis ticks

        if filename:
            plt.savefig(filename, dpi=300)  # Saving the interpolated images if a filename is provided
            
        if title:
            fig.suptitle(title)  # Adding a title to the visualization

        plt.show()  # Displaying the plot

    def plot_encoding(self, idx=0, title="", filename=""):
        """
        Plots the encoding of an input image.

        Args:
            idx (int): The index of the input image to plot the encoding (default: 0).
            title (str): The title of the visualization (optional).
            filename (str): The filename to save the plot (optional).
        """
        cell = self.data[idx]  # Selecting an input image
        if self.num_channels == 1:
            cell = cell.reshape(68, 68, 3)[:, :, self.channel]
        
        z = self.model.encode(cell.flatten())  # Encoding the input image to obtain latent vectors
        plt.bar(np.arange(self.latent_size), z.flatten())  # Plotting the bar chart of the latent vectors

        if filename:
            plt.savefig(filename, dpi=300)  # Saving the plot if a filename is provided

        if title:
            plt.suptitle(title)  # Adding a title to the visualization

        plt.show()  # Displaying the plot
        return z
        
    def find_interesting_features(self):
        """
        Finds interesting features by calculating cosine similarity.

        Returns:
            sim_scores (numpy.ndarray): Array of similarity scores for each feature.
        """
        sim_scores = np.zeros(self.latent_size)  # Initializing an array for similarity scores
        space = torch.linspace(-4, 4, 10)  # Creating a range of values to investigate each feature
        for i in tqdm(range(self.latent_size)):
            zs = torch.zeros(11, self.latent_size)  # Creating a tensor for latent vectors
            zs[1:, i] = space  # Setting the feature values to investigate
            decodings = self.model.decode(zs)  # Decoding the latent vectors to generate images
            sim = cosine_similarity(decodings[0, :].reshape(1, -1), decodings[1:, :])  # Calculating cosine similarity
            sim_scores[i] = np.mean(sim)  # Averaging the similarity scores for each feature

        return sim_scores

