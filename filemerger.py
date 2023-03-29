import os
old_path = "your path here for old files"
new_path = "your path here for new files"
os.chdir(new_path)
idx = 0
number_of_files = 480000

for subdir, dirs, files in os.walk(old_path):
    for file in files:
        origin = subdir + "/" + file
        shutil.move(origin, new_path)
        os.rename(new_path + file, new_path + str(idx))
        idx+=1
        print(idx / number_of_files * 100, "%")