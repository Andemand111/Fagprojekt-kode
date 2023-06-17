#%
from models import ClassifyNN, VAE
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, Dataset, random_split, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import torch
import numpy as np
import matplotlib.pyplot as plt
#%%
### loads data and the labels pertaining thereto
X = torch.load("data_matrix_encodings_normal")
X = X[:, :64]
y = np.load("moa_int_label.npy")
y -= 1

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=69)
#%%
model = ClassifyNN(64,512,"relu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_, counts = np.unique(y, return_counts=True)
label_weights = 1 / counts

def get_weighted_dataloader(dataset, labels, label_weights, num_samples = None):
    if num_samples == None:
        num_samples = len(labels)
    
    weights = np.array([label_weights[int(i)] for i in labels])
    train_sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    dataloader = DataLoader(dataset, batch_size=128, sampler=train_sampler, drop_last=True)
    return dataloader

class Data(Dataset):
    def __init__(self, data, labels, device):
        self.data = data
        self.labels = labels
        self.device = device
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.data[idx].to(self.device)
        y = torch.tensor(self.labels[idx]).to(self.device)
        return X, y

dataset = Data(X, y, device)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_labels, test_labels = y[train_dataset.indices], y[test_dataset.indices]
train_dataloader = get_weighted_dataloader(train_dataset, train_labels, label_weights)
test_dataloader = get_weighted_dataloader(test_dataset, test_labels, label_weights)
model.train(20, train_dataloader, verbose=0)
acc = model.test(test_dataloader)
print(acc)
#%%
vae = VAE(64)
vae.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/betargb/rgb_model_beta64")
x_hat = vae.decode(X[:10])

#%%
plt.imshow(x_hat[0].reshape(68,68,3))