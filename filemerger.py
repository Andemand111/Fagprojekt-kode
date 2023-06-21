# This script is used to merge the files from the singh pipeline into one folder.

import os
old_path = "your path here for old files"
new_path = "your path here for new files"
import shutil
old_path = "C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Data/singlecell/singh_cp_pipeline_singlecell_images/"
new_path = "C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Data/singlecell/singh_cp_pipeline_singlecell_images/merged_files/"
os.chdir(new_path)
idx = 0
number_of_files = 480000

for subdir, dirs, files in os.walk(old_path):
    for file in files:
        for subdir, dirs, files in os.walk(old_path):
            shutil.move(origin, new_path)
            os.rename(new_path + file, new_path + str(idx))
            idx+=1
            print(idx / number_of_files * 100, "%")

