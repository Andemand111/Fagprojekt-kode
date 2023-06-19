import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from models import VAE

# Load data and models

path = "G:/Mit drev/Project Splinter Cell/"
metadata = pd.read_csv(path + "metadata.csv")
old_path = "C:/Users/Andba/Desktop/singlecell/singh_cp_pipeline_singlecell_images/"
new_path = "C:/Users/Andba/Desktop/encodings/"

rgb_model_paths = "G:/Mit drev/Project Splinter Cell/modeller/rgbmodeller/"
single_model_paths = "G:/Mit drev/Project Splinter Cell/modeller/singlechannelmodeller/"

# Create encodings
for typ in ["beta", "normal"]:
    encodings = torch.zeros(len(metadata), 160)
    
    rgb_model = VAE(64)
    r_model = VAE(32, num_channels=1)
    g_model = VAE(32, num_channels=1)
    b_model = VAE(32, num_channels=1)
    
    rgb_model.load_model(rgb_model_paths + f"rgb_model_{typ}64")
    r_model.load_model(single_model_paths + f"r_model_{typ}32")
    g_model.load_model(single_model_paths + f"g_model_{typ}32")
    b_model.load_model(single_model_paths + f"b_model_{typ}32")

    for i in tqdm(range(len(metadata))):
        folder_name = metadata["Multi_Cell_Image_Name"][i] + "/"
        file_name = metadata["Single_Cell_Image_Name"][i]
        file = old_path + folder_name + file_name
        
        cell = np.load(file).astype(np.float64).reshape(68, 68, 3)
        cell /= np.amax(cell, axis=(0, 1))
        cell = torch.from_numpy(cell).float()

        enc1 = rgb_model.encode(cell).flatten()
        enc2 = r_model.encode(cell[:, :, 0]).flatten()
        enc3 = g_model.encode(cell[:, :, 1]).flatten()
        enc4 = b_model.encode(cell[:, :, 2]).flatten()
        
        encoding = torch.cat((enc1, enc2, enc3, enc4))
        encodings[i, :] = encoding
    
    torch.save(encodings, new_path + f"{typ}_encodings")
        