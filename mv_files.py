import os
DATA_DIR = "/home/blackfoot/codes/Object-Graph-Memory/IL_data/pcl_gibson"
files = os.listdir(DATA_DIR)
files = [file for file in files if "train" not in file]
files = [file for file in files if "val" not in file]
print("aa")
for i, file_path in enumerate(files):
    os.rename(os.path.join(DATA_DIR, file_path), os.path.join(DATA_DIR, "train", file_path))
    print(f"[{i}/{len(files)}] moved", file_path)