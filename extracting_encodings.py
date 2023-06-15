from models import VAE
import numpy as np
import torch
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
latent_size = 64 + 32*3
latent_sizergb = 64
latent_sizesingle = 32

path = "C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Data/singlecell/singh_cp_pipeline_singlecell_images/merged_files/"


#%%
""" Normal-VAE models""" 
vaergb = VAE(latent_sizergb).to(device)
vaergb.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/normalrgb/rgb_model_normal64")

vaer = VAE(latent_sizesingle, num_channels=1).to(device)
vaer.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/normalsinglecolor/r_model_normal32")

vaeg = VAE(latent_sizesingle, num_channels=1).to(device)
vaeg.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/normalsinglecolor/g_model_normal32")

vaeb = VAE(latent_sizesingle, num_channels=1).to(device)
vaeb.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/normalsinglecolor/b_model_normal32")

#%%
""" Beta-VAE models""" 
vaergb = VAE(latent_sizergb).to(device)
vaergb.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/betargb/rgb_model_beta64")

vaer = VAE(latent_sizesingle, num_channels=1).to(device)
vaer.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/betasinglecolor/r_model_beta32")

vaeg = VAE(latent_sizesingle, num_channels=1).to(device)
vaeg.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/betasinglecolor/g_model_beta32")

vaeb = VAE(latent_sizesingle, num_channels=1).to(device)
vaeb.load_model("C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/FærdigeModeller/betasinglecolor/b_model_beta32")

#%%
# Normal encodings
indxs = np.load("moa_indices.npy")
labels = np.load("moa_int_label.npy")
labels -= 1

X_normal = torch.zeros((len(labels), latent_size))

for i, idx in tqdm(enumerate(indxs)):
    pic = np.load(path + str(idx)).astype(np.float32).reshape(68,68,3)
    pic /= np.amax(pic, axis=(0, 1))
    
    pic = torch.from_numpy(pic).to(device)
    
    enc1 = vaergb.encode(pic.view(68,68,3)).flatten()
    enc2 = vaer.encode(pic[:, :, 0]).flatten()
    enc3 = vaeg.encode(pic[:, :, 1]).flatten()
    enc4 = vaeb.encode(pic[:, :, 2]).flatten()

    features = torch.cat((enc1, enc2, enc3, enc4))
    X_normal[i, :] = features
    
torch.save(X_normal, "data_matrix_encodings_normal")

#%%
# Beta encodings
indxs = np.load("moa_indices.npy")
labels = np.load("moa_int_label.npy")
labels -= 1

X_beta = torch.zeros((len(labels), latent_size))

for i, idx in tqdm(enumerate(indxs)):
    pic = np.load(path + str(idx)).astype(np.float32).reshape(68,68,3)
    pic /= np.amax(pic, axis=(0, 1))
    
    pic = torch.from_numpy(pic).to(device)
    
    enc1 = vaergb.encode(pic.view(68,68,3)).flatten()
    enc2 = vaer.encode(pic[:, :, 0]).flatten()
    enc3 = vaeg.encode(pic[:, :, 1]).flatten()
    enc4 = vaeb.encode(pic[:, :, 2]).flatten()

    features = torch.cat((enc1, enc2, enc3, enc4))
    X_beta[i, :] = features


torch.save(X_beta, "data_matrix_encodings_beta")

#%%