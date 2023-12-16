import os
from pathlib import Path
from tqdm import tqdm

tar_folder = "/media/shij3/T7/LoDoInd/Noise1"
test_ratio = 0.5

path_parts = os.path.split(tar_folder)
tar_ref_folder = os.path.join(*path_parts[:-1], "Ref")

files = os.listdir(tar_folder)
files.sort()

test_folder_path = os.path.join(*path_parts[:-1], path_parts[-1]+"_test")
test_ref_folder_path = os.path.join(*path_parts[:-1], "Ref_test")

# Create the new folder if it does not exist
print(f"Creating {test_folder_path}")
if not os.path.exists(test_folder_path):
    Path(test_folder_path).mkdir(exist_ok=True)

print(f"Creating {test_ref_folder_path}")
if not os.path.exists(test_ref_folder_path):
    Path(test_ref_folder_path).mkdir(exist_ok=True)

# Move the files
start_index = int(len(files) * test_ratio)
for file in tqdm(files[start_index:]):
    os.rename(os.path.join(tar_folder, file), os.path.join(test_folder_path, file))
    os.rename(os.path.join(tar_ref_folder, file), os.path.join(test_ref_folder_path, file))