from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import torch
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

distribution = "beta"
random_state = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
### loads data and the labels pertaining thereto
data_path = "G:/Mit drev/Uni/4. semester/fagprojekt/fra_git/"
data = torch.load(data_path + f"{distribution}_encodings")
indxs = np.load(data_path + "moa_indices.npy")
labels = np.load("moa_int_label.npy")
labels = labels - 1
data = data[indxs]

sampler = RandomOverSampler(sampling_strategy = "auto")
# sampler = RandomUnderSampler(sampling_strategy = "auto")
X, y = sampler.fit_resample(data, labels)

n_neighbors = [1, 5, 10, 20, 40]

## makes folds
k1 = k2 = 5
skf_outer = StratifiedKFold(n_splits=k1, shuffle=True, random_state=random_state)
skf_inner = StratifiedKFold(n_splits=k2, shuffle=True, random_state=random_state)

results = np.zeros((k1, 2))

for outer_fold, (D_par_index, D_test_index) in enumerate(skf_outer.split(X, y)):
    D_par = X[D_par_index]
    D_par_labels = y[D_par_index]
    
    D_test = X[D_test_index]
    D_test_labels = y[D_test_index]
    
    accs = np.zeros((k2, len(n_neighbors)))
    
    for inner_fold, (D_train_index, D_val_index) in enumerate(skf_inner.split(D_par, D_par_labels)):
        D_train = D_par[D_train_index]
        D_train_labels = D_par_labels[D_train_index]
        
        D_val = D_par[D_val_index]
        D_val_labels = D_par_labels[D_val_index]
        
        for s, n_neig in enumerate(n_neighbors):
            print(f"Outer: {outer_fold}, inner: {inner_fold}, model: {s}")
            print("training..")
            model = KNeighborsClassifier(n_neighbors=n_neig)
            model.fit(D_train, D_train_labels)
            acc = model.score(D_val, D_val_labels)
            accs[inner_fold, s] = acc
            print(f"acc = {acc}")
            print("..done!")
            
    E_s_acc = np.mean(accs, 0)
    best_model_index = np.argmax(E_s_acc)
    results[outer_fold, 0] = best_model_index 
    best_parameter = n_neighbors[best_model_index]
    best_model = KNeighborsClassifier(n_neighbors=best_parameter)
    best_model.fit(D_par, D_par_labels)
    results[outer_fold, 1] = best_model.score(D_test, D_test_labels)
        
E_gen = np.mean(results[:, 1])
np.save(f"two_level_test_{distribution}", results)
print(E_gen * 100, "%")