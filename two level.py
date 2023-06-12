from models import VAE, ClassifyNN
from dataset import ClassifyCells
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

import torch
import numpy as np

device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")

latent_size = 700
vae = VAE(latent_size).to(device)
vae.load_model("your_path")
indxs = np.load("your_path")
labels = np.load("your_path")
labels -= 1

_, counts = np.unique(labels, return_counts=True)
label_weights = 1 / counts

dataset = ClassifyCells("your_path", [vae], [None], indxs, labels, device)

k1 = k2 = 5
skf_outer = StratifiedKFold(n_splits=k1, shuffle=True, random_state=69)
skf_inner = StratifiedKFold(n_splits=k2, shuffle=True, random_state=69)

num_samples = 300
num_epochs = 1

def get_weighted_dataloader(dataset, labels, label_weights, num_samples = None):
    if num_samples == None:
        num_samples = len(labels)
    
    weights = np.array([label_weights[int(i)] for i in labels])
    train_sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    dataloader = DataLoader(dataset, batch_size=128, sampler=train_sampler, drop_last=True)
    return dataloader
    
def make_new_models(num_hiddens, activations):
    models = []
    for num_hidden, act_func in zip(num_hiddens, activations):
        model = ClassifyNN(latent_size, num_hidden, act_func)
        models.append(model)
    return models

num_hiddens = [1, 32, 64 ,128, 256]
activations = ["identity", "relu", "relu", "relu", "relu"]

models = make_new_models(num_hiddens, activations)
E_gens = np.zeros(k1)

for outer_fold, (D_par_index, D_test_index) in enumerate(skf_outer(dataset, labels)):    
    D_par = Subset(dataset, D_par_index)
    D_test = Subset(dataset, D_test_index)
    
    D_par_labels = labels[D_par_index]
    D_test_labels = labels[D_test_index]
    
    for inner_fold, (D_train_index, D_val_index) in enumerate(skf_inner(D_par, D_par_labels)):        
        accs = np.zeros((k2, len(models)))
        
        D_train = Subset(D_par, D_train_index)
        D_val = Subset(D_par, D_val_index)
        
        train_labels = D_par_labels[D_train_index]
        val_labels = D_val[D_val_index]
        
        train_dataloader = get_weighted_dataloader(D_train, train_labels, label_weights, num_samples)
        val_dataloader = get_weighted_dataloader(D_val, val_labels, label_weights, num_samples)
        
        for s, model in enumerate(models):
            print(f"Outer: {outer_fold}, inner: {inner_fold}, model: {s}")
            print("training..")
            
            model.train(num_epochs, train_dataloader, verbose=0)
            acc = model.test(val_dataloader)
            accs[inner_fold, s] = acc
            
            print("..done!")
        
    E_s_acc = np.mean(accs, 0)
    best_model_index = np.argmax(E_s_acc)
    best_parameters = (latent_size, num_hiddens[best_model_index], activations[best_model_index])
    best_model = ClassifyNN(*best_parameters)
    
    par_dataloader = get_weighted_dataloader(D_par, D_par_labels, label_weights, num_samples)
    test_dataloader = get_weighted_dataloader(D_test, D_test_labels, label_weights, num_samples)
    
    print(f"Training and testing best model in outer: {outer_fold}")
    best_model.train(num_epochs, par_dataloader, verbose=0)
    E_gens[outer_fold] = best_model.test(test_dataloader)
    best_model.save(f"best_model{outer_fold}")
    print("Done!")
    
    models = make_new_models(num_hiddens, activations)
    
E_gen = np.mean(E_gens)
np.save("E_gens", E_gens)
print(E_gen * 100, "%")