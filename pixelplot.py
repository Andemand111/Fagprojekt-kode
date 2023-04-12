import matplotlib.pyplot as plt
import numpy as np

data = np.load("din filvej her")

data = data.reshape(-1,3)
datamax = np.max(data,axis=0)
data = data/datamax

#%%


plt.subplot(2,2,1)
plt.hist(data[:,0],bins = 50)

plt.subplot(2,2,2)
plt.hist(data[:,1],bins = 50)

plt.subplot(2,2,3)
plt.hist(data[:,2],bins = 50)

plt.show()