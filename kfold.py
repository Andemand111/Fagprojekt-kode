import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedKFold
from dataset import SmallCells
from models import ClassifyNN
import numpy as np

# %%
data = np.load("celle_data.npy")
ps = [0.05, 0.1, 0.2, 0.05, 0.1, 0.05, 0.05, 0.1, 0.05, 0.1, 0.1, 0.05]
labels = np.random.choice(np.arange(12), size=20000, p=ps)
dataset = SmallCells(data)

# %%

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
        weights, num_samples=len(test_labels), replacement=True)

    train_dataloader = DataLoader(
        train_subset, batch_size=128, sampler=train_sampler, drop_last=True)
    test_dataloader = DataLoader(
        test_subset, batch_size=len(test_subset), sampler=test_sampler)

    model = ClassifyNN(512)
    model.train(50, train_dataloader)
    loss = model.test(test_dataloader)
    losses[i] = loss

print(losses.mean())
