import pandas as pd
import os
import shutil
from tqdm import tqdm

path = "G:/Mit drev/Project Splinter Cell/"
metadata = pd.read_csv(path + "metadata.csv")
old_path = "C:/Users/Andba/Desktop/singlecell/singh_cp_pipeline_singlecell_images/"
new_path = "C:/Users/Andba/Desktop/cells/"

for i in tqdm(range(len(metadata))):
    folder_name = metadata["Multi_Cell_Image_Name"][i] + "/"
    file_name = metadata["Single_Cell_Image_Name"][i]
    file = old_path + folder_name + file_name
    shutil.copy(file, new_path)
    os.rename(new_path + file_name, new_path + "i")
