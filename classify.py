from models import ClassifyNN, VAE
from dataset import ClassifyCells
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

import numpy as np

rgb_model = VAE(700)
r_model = VAE(100, num_channels=1)
g_model = VAE(100, num_channels=1)
b_model = VAE(100, num_channels=1)

rgb_model.load_model("rgb_model")
r_model.load_model("r_model")
g_model.load_model("g_model")
b_model.load_model("b_model")

path = "C:/Users/Andba/Desktop/Celle data/"
indxs = np.load("moa_indices.npy")
labels = np.load("moa_int_label.npy") - 1

unique_values, counts = np.unique(labels, return_counts=True)
label_weights = 1 / counts
weights = [label_weights[int(i)] for i in labels]

sampler = WeightedRandomSampler(weights, num_samples = len(weights), replacement=True)
dataset = ClassifyCells(path, [rgb_model, r_model, g_model, b_model], indxs, labels)
train_data, test_data, val_data = random_split(dataset, [0.9, 0.09, 0.01])

kwargs = dict(batch_size=128, drop_last=True, shuffle=True)
train_dataloader = DataLoader(train_data, **kwargs)

model = ClassifyNN(1000)
stats = model.train(50, train_dataloader, val_data=val_data)
test_accuracy = model.test(test_data)