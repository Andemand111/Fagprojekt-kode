from models import ClassifyNN
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, Dataset
import torch
import numpy as np
#from tqdm import tqdm

num_samples = None             ## how much data is used. None = all data
num_epochs = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
latent_sizergb = 64
latent_sizesingle = 32
latent_size = latent_sizergb + latent_sizesingle * 3   ## total size of model input using all 4 models

### loads data and the labels pertaining thereto
data = torch.load("data_matrix_encodings_beta")
labels = np.load("moa_int_label.npy")
labels -= 1

### defines the dataset class and makes an instance of it
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

dataset = Data(data, labels, device)

## calculates label_weights which are later used for making a weighted sampler
_, counts = np.unique(labels, return_counts=True)
label_weights = 1 / counts

## makes folds
k1 = k2 = 5
skf_outer = StratifiedKFold(n_splits=k1, shuffle=True, random_state=69)
skf_inner = StratifiedKFold(n_splits=k2, shuffle=True, random_state=69)

## functions used for getting weighted dataloader and new models
def get_weighted_dataloader(dataset, labels, label_weights, num_samples = None):
    if num_samples == None:
        num_samples = len(labels)
    
    weights = np.array([label_weights[int(i)] for i in labels])
    train_sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    dataloader = DataLoader(dataset, batch_size=128, sampler=train_sampler, drop_last=True)
    return dataloader
    
def make_new_models(num_hiddens, activations, device):
    models = []
    for num_hidden, act_func in zip(num_hiddens, activations):
        model = ClassifyNN(latent_size, num_hidden, act_func).to(device)
        models.append(model)
    return models

## make new models
num_hiddens = [256, 128, 256, 512, 1024]
activations = ["relu", "relu", "relu", "relu", "relu"]
models = make_new_models(num_hiddens, activations, device)

results = np.zeros((k1, 2))


## k-fold cross validation loop
for outer_fold, (D_par_index, D_test_index) in enumerate(skf_outer.split(dataset, labels)):    
    D_par = Subset(dataset, D_par_index)
    D_test = Subset(dataset, D_test_index)
    
    D_par_labels = labels[D_par_index]
    D_test_labels = labels[D_test_index]
    accs = np.zeros((k2, len(models)))

    
    
    for inner_fold, (D_train_index, D_val_index) in enumerate(skf_inner.split(D_par, D_par_labels)):        
        
        D_train = Subset(D_par, D_train_index)
        D_val = Subset(D_par, D_val_index)
        
        train_labels = D_par_labels[D_train_index]
        val_labels = D_par_labels[D_val_index]
        
        train_dataloader = get_weighted_dataloader(D_train, train_labels, label_weights, num_samples)
        val_dataloader = get_weighted_dataloader(D_val, val_labels, label_weights, num_samples)
        
        for s, model in enumerate(models):
            print(f"Outer: {outer_fold}, inner: {inner_fold}, model: {s}")
            print("training..")
            
            model.train(num_epochs, train_dataloader, verbose=0)
            acc = model.test(val_dataloader)
            accs[inner_fold, s] = acc
            print(acc * 100, "%")
            
            print("..done!")


        
    E_s_acc = np.mean(accs, 0)
    best_model_index = np.argmax(E_s_acc)
    results[outer_fold, 0] = best_model_index 
    best_parameters = (latent_size, num_hiddens[best_model_index], activations[best_model_index])
    best_model = ClassifyNN(*best_parameters).to(device)
    
    par_dataloader = get_weighted_dataloader(D_par, D_par_labels, label_weights, num_samples)
    test_dataloader = get_weighted_dataloader(D_test, D_test_labels, label_weights, num_samples)
    
    print(f"Training and testing best model in outer: {outer_fold}")
    best_model.train(num_epochs, par_dataloader, verbose=0)
    results[outer_fold, 1] = best_model.test(test_dataloader)
    best_model.save_model(f"best_model{outer_fold}_beta")
    print("Done!")
    
    models = make_new_models(num_hiddens, activations, device)
    
E_gen = np.mean(results[:, 1])
np.save("results_beta", results)
print(E_gen * 100, "%")

