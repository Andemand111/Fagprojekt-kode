#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

metadata = pd.read_csv("metadata.csv")

#%%
metadata.head()
# %%
# count the number of entries in moa column that are not DMSO
metadata["moa"].value_counts()
#%%
### make a new list that are integer labels for all values in "moa", set DMSO to 0###
moa = metadata["moa"].to_numpy()
### make labels for all unique moa values, set DMSO to 0 ###
moa_labels = np.unique(moa)
moa_labels = np.delete(moa_labels, np.where(moa_labels == "DMSO"))
#%%
# integer labelled moa values
moa_int = np.zeros(len(moa))
for i in range(len(moa)):
    if moa[i] != "DMSO":
        moa_int[i] = np.where(moa_labels == moa[i])[0][0] + 1

#%%
### List of all indices where moa is not 0 in moa_int ###
moa_indices = np.where(moa_int != 0)[0]
#%%

