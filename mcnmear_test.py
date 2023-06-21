from sklearn.neighbors import KNeighborsClassifier
import torch
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from mlxtend.evaluate import mcnemar_table, mcnemar

# %%
# Load data

random_state = 42
np.random.random_state = random_state
data_path = "G:/Mit drev/Uni/4. semester/fagprojekt/fra_git/"
indxs = np.load(data_path + "moa_indices.npy")
labels = np.load("moa_int_label.npy")
labels = labels - 1
data_beta = torch.load(data_path + "beta_encodings")[indxs]
data_normal = torch.load(data_path + "normal_encodings")[indxs]

# %%
# Split data

sampler = RandomUnderSampler(sampling_strategy = "auto", random_state = random_state)
X_beta, y = sampler.fit_resample(data_beta, labels)
X_normal, y  = sampler.fit_resample(data_normal, labels)

train_size = int(0.8 * len(X_beta))
arange = np.arange(len(X_beta))
np.random.shuffle(arange)
train_indxs = arange[:train_size]
test_indxs = arange[train_size:]

X_beta_train = X_beta[train_indxs]
X_normal_train = X_normal[train_indxs]
y_train = y[train_indxs]

X_beta_test = X_beta[test_indxs]
X_normal_test = X_normal[test_indxs]
y_test = y[test_indxs]

# %%
# Train and test models

beta_model = KNeighborsClassifier(n_neighbors=20)
normal_model = KNeighborsClassifier(n_neighbors=20)

beta_model.fit(X_beta_train,  y_train)
normal_model.fit(X_normal_train, y_train)

beta_preds = beta_model.predict(X_beta_test)
normal_preds = normal_model.predict(X_normal_test)


#%%
# McNemar test
table = mcnemar_table(y_target=y_test, 
                      y_model1=beta_preds, 
                      y_model2=normal_preds)

result = mcnemar(ary=table)
print("McNemar test statistic:", result[0])
print("p-value:", result[1])