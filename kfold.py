#%%
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedKFold
from dataset import ClassifyCells, SmallCells
from models import ClassifyNN, VAE
import numpy as np

#%%

vae = VAE(700)
vae.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/betamodels/rgb_model_beta")
#%%
path = "C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Data/singlecell/singh_cp_pipeline_singlecell_images/merged_files/"
indxs = np.load("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Fagprojekt-kode/moa_indices.npy")
labels = np.load("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Fagprojekt-kode/moa_int_label.npy")
labels -= 1
dataset = ClassifyCells(path, [vae], color_channels = [None], indxs=indxs, labels=labels)

#%%

k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=69)
losses = np.zeros(k)

for i, (train_index, test_index) in enumerate(skf.split(dataset, labels)):
    train_labels, test_labels = labels[train_index], labels[test_index]
    train_subset, test_subset = Subset(
        dataset, train_index), Subset(dataset, test_index)

    _, counts = np.unique(train_labels, return_counts=True)
    label_weights = 1 / counts
    weights = np.array([label_weights[int(i)] for i in train_labels])
    train_sampler = WeightedRandomSampler(
        weights, num_samples=len(train_labels), replacement=True)

    _, counts = np.unique(test_labels, return_counts=True)
    label_weights = 1 / counts
    weights = np.array([label_weights[int(i)] for i in test_labels])
    test_sampler = WeightedRandomSampler(
        weights, num_samples=2000, replacement=True)

    train_dataloader = DataLoader(
        train_subset, batch_size=128, sampler=train_sampler, drop_last=True)
    test_dataloader = DataLoader(
        test_subset, batch_size=128, sampler=test_sampler, drop_last=True)

    model = ClassifyNN(700)
    model.train(10, train_dataloader)
    loss = model.test(test_dataloader)
    losses[i] = loss

print(losses.mean())
