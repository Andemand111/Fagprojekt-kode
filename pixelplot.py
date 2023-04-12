import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from scipy.stats import norm
from fitter import Fitter
#%%

data = np.load("celle_data.npy")
#%%

rgbmax = np.amax(data.reshape(-1, 68, 68, 3), axis=(1,2), keepdims=True)
normalized_data = data.reshape(-1,68,68,3) / rgbmax

#%%
np.random.seed(1)
eps = 1e-3
fig, axs = plt.subplots(3,3)
for ax in axs.flatten():
    randx = np.random.randint(68)
    randy = np.random.randint(68)
    randc = np.random.randint(3)
    
    pixel = normalized_data[:, randx, randy, randc].flatten()
    # ax.hist(pixel, density=True, bins=50)
    
    # ax.set_xticks([])
    # ax.set_yticks([])
    
    # a,b,_,_ = beta.fit(np.clip(pixel, eps, 1-eps))    
    # xs = np.linspace(eps, 1-eps)
    # ys = beta.pdf(xs, a, b)
    # ax.plot(xs, ys)
    

    # mu, std = norm.fit(pixel)
    # xs = np.linspace(0,1)
    # ys = norm.pdf(xs, mu, std)
    # ax.plot(xs, ys)
    
    f = Fitter(pixel,
           distributions=["beta",
                          "norm"])
    f.fit()
    f.summary()
    plt.show()