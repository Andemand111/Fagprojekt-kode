from models import ClassifyNN, VAE
from dataset import ClassifyCells
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

import numpy as np

rgb_model = VAE(700)
r_model = VAE(100, num_channels=1)
g_model = VAE(100, num_channels=1)
b_model = VAE(100, num_channels=1)

rgb_model.load_model("rgb_model_normal")
r_model.load_model("r_model")
g_model.load_model("g_model")
b_model.load_model("b_model")

path = "C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Data/singlecell/singh_cp_pipeline_singlecell_images/merged_files/"  # path to data
indxs = np.load("moa_indices.npy")
labels = np.load("moa_int_label.npy") - 1

dataset = ClassifyCells(path, [rgb_model], indxs, labels)
train_data, test_data, val_data = random_split(dataset, [0.95, 0.04, 0.01])

train_labels = labels[train_data.indices]
unique_values, counts = np.unique(train_labels, return_counts=True)
label_weights = 1 / counts
weights = [label_weights[int(i)] for i in train_labels]
sampler = WeightedRandomSampler(weights, num_samples=20000, replacement=True)

kwargs = dict(batch_size=128, drop_last=True, sampler = sampler)
train_dataloader = DataLoader(train_data, **kwargs)

model = ClassifyNN(700)
stats = model.train(20, train_dataloader, val_data=val_data)
test_accuracy = model.test(test_data)
print(test_accuracy * 100, "%")
