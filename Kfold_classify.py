import torch
import numpy as np
from dataset import ClassifyCells
from models import VAE, ClassifyNN
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, random_split
from sklearn.model_selection import KFold

# Initialize parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
latent_size = 64 + 32*3
latent_sizergb = 64
latent_sizesingle = 32

# Load data and models

vaergb = VAE(latent_sizergb).to(device)
vaergb.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/normalrgb/rgb_model_normal64")

vaer = VAE(latent_sizesingle, num_channels=1).to(device)
vaer.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/normalsinglecolor/r_model_normal32")

vaeg = VAE(latent_sizesingle, num_channels=1).to(device)
vaeg.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/normalsinglecolor/g_model_normal32")

vaeb = VAE(latent_sizesingle, num_channels=1).to(device)
vaeb.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/normalsinglecolor/b_model_normal32")

# Load datalabels and indices

indxs = np.load("moa_indices.npy")
labels = np.load("moa_int_label.npy")
labels -= 1
dataset = ClassifyCells("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Data/singlecell/singh_cp_pipeline_singlecell_images/merged_files/", [vaergb,vaer,vaeg,vaeb], [None,0,1,2], indxs, labels)

# Make dataloaders

_, counts = np.unique(labels, return_counts=True)
label_weights = 1 / counts

def get_weighted_dataloader(dataset, labels, label_weights, num_samples = None):
    if num_samples == None:
        num_samples = len(labels)
    
    weights = np.array([label_weights[int(i)] for i in labels])
    train_sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    dataloader = DataLoader(dataset, batch_size=num_samples, sampler=train_sampler, drop_last=True)
    return dataloader

# Make models

def make_new_models(num_hiddens, activations):
    models = []
    for num_hidden, act_func in zip(num_hiddens, activations):
        model = ClassifyNN(latent_size, num_hidden, act_func)
        models.append(model)
    return models

num_hiddens = [1,128, 256, 512, 1024]
activations = ["identity", "relu", "relu", "relu", "relu"]

models = make_new_models(num_hiddens, activations)

dataloader = get_weighted_dataloader(dataset, labels, label_weights, num_samples = 1000)
X, y = None, None
for X_, y_ in dataloader:
    X = X_
    y = y_
    
kf_outer = KFold(n_splits = 5, shuffle=False, random_state = 69)
kf_inner = KFold(n_splits = 5, shuffle=False, random_state = 69)

results = np.zeros((5, 2))

# Outer loop

for i, (train_index, test_index) in enumerate(kf_outer(X)):
    D_par, D_par_labels = X[train_index], y[train_index]
    D_test, D_test_labels = X[test_index], y[test_index]
    
    # Inner loop
    
    for j, (D_train_index, D_val_index) in enumerate(kf_inner(D_par)):
        D_train, D_train_labels = D_par[D_train_index], D_par_labels[D_train_index]
        D_val, D_val_labels = D_par[D_val_index], D_test_labels[D_val_index]