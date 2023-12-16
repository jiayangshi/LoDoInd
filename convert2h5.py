import os
import h5py
import numpy as np
from tifffile import imread
from tqdm import tqdm

train_folder = '/media/shij3/T7/LoDoInd/Noise1'
test_folder = train_folder + '_test'
path_parts = os.path.split(train_folder)
ref_folder = os.path.join(*path_parts[:-1], "Ref")
ref_test_folder = ref_folder + '_test'

folders = [train_folder, test_folder, ref_folder, ref_test_folder]

train_slices = len(os.listdir(train_folder))
test_slices = len(os.listdir(test_folder))

train_h5 = os.path.join(*path_parts[:-1], path_parts[-1]+".h5")
train_ref_h5 = os.path.join(*path_parts[:-1], "Ref.h5")
test_h5 = os.path.join(*path_parts[:-1], path_parts[-1]+"_test.h5")
test_ref_h5 = os.path.join(*path_parts[:-1], "Ref_test.h5")

h5s = [train_h5, train_ref_h5, test_h5, test_ref_h5]

for i in range(len(folders)):
    h5 = h5s[i]
    f = h5py.File(h5, 'w')
    folder = folders[i]
    dataset = f.create_dataset('data', shape=(len(os.listdir(folder)),1250,1250), dtype=np.float32)    

    files = os.listdir(folder)
    files.sort()
    for j in tqdm(range(len(files))):
        img = imread(os.path.join(folder, files[j])).astype(np.float32)
        dataset[j,:,:] = img