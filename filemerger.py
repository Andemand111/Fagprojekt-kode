import os
import shutil
old_path = "C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Data/singlecell/singh_cp_pipeline_singlecell_images/"
new_path = "C:/Users/gusta/OneDrive/Skrivebord/KI & Data/Semester 4/Fagprojekt/Data/singlecell/singh_cp_pipeline_singlecell_images/merged_files/"
os.chdir(new_path)
idx = 0
number_of_files = 488000

for subdir, dirs, files in os.walk(old_path):
    for file in files:
        origin = subdir + "/" + file
        shutil.move(origin, new_path)
        os.rename(new_path + file, new_path + str(idx))
        idx+=1
        print(idx / number_of_files * 100, "%")
