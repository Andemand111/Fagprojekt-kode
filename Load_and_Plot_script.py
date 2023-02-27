import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 300)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import cv2

# Code by Gustav Rørhauge & Andreas Bagge

import os
# Set working directory
os.chdir("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Data/singlecell/singh_cp_pipeline_singlecell_images/B02_s1_w1B1A7ADEA-8896-4C7D-8C63-663265374B72")


# Open a file
path = "C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Data/singlecell/singh_cp_pipeline_singlecell_images/B02_s1_w1B1A7ADEA-8896-4C7D-8C63-663265374B72"
dirs = os.listdir( path )

# Load all files from 1 folder
arrays = []
for filename in dirs:
    flatfile = np.load(filename).flatten()
    arrays.append(list(flatfile/np.max(flatfile)))
arrays = np.asarray(arrays)




#%%
# Plot Function
pic1 = arrays[1][:]
def showim(vec, brighten=False, ax=False):
    if brighten:
        fac = 1 / np.max(vec)
        vec *= fac
    
    im = np.reshape(vec, (68,68,3))
    if ax:
        ax.imshow(im)
    else:
        plt.imshow(im)
    

showim(pic1)
showim(pic1, brighten = True)


#%%
#Plot random pictures

numberofimages = np.random.randint(1,arrays.shape[0],9)
fig,axs = plt.subplots(3,3)

for i in range(9):
    x = i // 3
    y = i % 3
    ax = axs[x,y]
    showim(arrays[numberofimages[i]][:], brighten = True, ax = ax)
    
