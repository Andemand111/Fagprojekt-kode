import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Code by Gustav Rørhauge & Andreas Bagge

import os
# Set working directory
os.chdir("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Data/singlecell/singh_cp_pipeline_singlecell_images/B02_s1_w1B1A7ADEA-8896-4C7D-8C63-663265374B72")


# Set a filepath
path = "C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Data/singlecell/singh_cp_pipeline_singlecell_images/"
dirs = os.listdir( path )
path2 = "C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Data/singlecell/singh_cp_pipeline_singlecell_images"
topfolder = os.listdir( path2 )


""""Dataloader"""

filer = 0
n = 20000
terminate = False

data = np.zeros((n, 13872))
#%%
for folder in topfolder:
    dirs = os.listdir(path + folder )
    for filename in dirs:
        print(filer / n)
        
        flatfile = np.load(path + folder + "/" + filename).flatten()
        data[filer, :] = flatfile
        
        filer += 1
        
        if filer == n:
            terminate = True
            break
            
    if terminate:
        break



#%%
# Plot Function
pic1 = data[1]
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

numberofimages = np.random.randint(1,data.shape[0],9)
fig,axs = plt.subplots(3,3)

for i in range(9):
    x = i // 3
    y = i % 3
    ax = axs[x,y]
    showim(data[numberofimages[i]][:], brighten = True, ax = ax)
    
    
    
#%%
meanpic = np.mean(data.reshape(-1,68,68,3),axis=0)    
#%%
showim(meanpic, brighten = True)
# %%

"""covariance, be careful! """
# cova = np.cov(data.T)
# #%%
# sns.heatmap(cova)
# #%%
# plt.imshow(cova,cmap = 'viridis')

#%% 
"""Summary statistics """ 
channel_len = int(data.shape[1]/3)

dadler = data.reshape(-1,3)
RGBstd = np.std(dadler,axis=0)

### Standard Deviations
# stdR = 2388.3 , stdG = 3638.45, stdB = 1912.5

#%%
lst = np.arange(2700)
print(f'arange: {lst}')
picex= lst.reshape(-1,30,30,3)
print(f'reshaped: {picex}')
