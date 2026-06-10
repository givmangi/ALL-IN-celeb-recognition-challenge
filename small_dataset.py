import os
import shutil

# 1. Define paths
KAGGLE_DATA_PATH = "C:\\Users\\mangi\\OneDrive\\Desktop\\sku\\ml_challenge\\ALL-IN-celeb-recognition-challenge\\dataset\\VGGface2_HQ\\full_dataset\\VGGface2_None_norm_512_true_bygfpgan"
TINY_DATA_PATH = "C:\\Users\\mangi\\OneDrive\\Desktop\\sku\\ml_challenge\\ALL-IN-celeb-recognition-challenge\\dataset\\tiny_dataset"

print("Creating a tiny dataset for fast testing...")

# 2. Create the temp directory (wipes it if it already exists)
if os.path.exists(TINY_DATA_PATH):
    shutil.rmtree(TINY_DATA_PATH)
os.makedirs(TINY_DATA_PATH)

all_folders = sorted(os.listdir(KAGGLE_DATA_PATH))
folders_to_copy = all_folders[:500] 

for folder in folders_to_copy:
    src_folder = os.path.join(KAGGLE_DATA_PATH, folder)
    dest_folder = os.path.join(TINY_DATA_PATH, folder)
    shutil.copytree(src_folder, dest_folder)