import numpy as np
from torch.utils.data import Dataset
import random
import os
from tifffile import imread
import h5py

class DatasetStackTIF(Dataset):
    def __init__(self, recon_low, recon_high, stack=5):
        self.recon_low = os.listdir(recon_low)
        self.recon_low.sort()
        self.recon_low = np.vectorize(lambda x, y: os.path.join(x,y))(recon_low, self.recon_low)
        self.recon_high = os.listdir(recon_high)
        self.recon_high.sort()
        self.recon_high = np.vectorize(lambda x, y: os.path.join(x,y))(recon_high, self.recon_high)
        assert stack%2 == 1 # stack can only be odd number eg. 1,3,5,7
        self.stack = stack

    def __len__(self):
        return len(self.recon_low)

    def __getitem__(self, idx):
        edge = self.stack // 2
        img_idxs = np.arange(idx - edge, idx + edge + 1)
        img_idxs[img_idxs < 0] = 0
        img_idxs[img_idxs > len(self.recon_low) - 1] = len(self.recon_low) - 1
        imgs = []

        for m in img_idxs:
            image = imread(self.recon_low[m]).astype(np.float32)
            image = np.expand_dims(image, (0))
            imgs.append(image)
        label = imread(self.recon_high[idx]).astype(np.float32)

        image = np.concatenate(imgs, axis=0)
        label = np.expand_dims(label, (0))

        return image, label
    

class DatasetPatch(Dataset):
    def __init__(self, recon_low, recon_high, patch_size=(16,1250,1250)):
        self.recon_low = recon_low
        self.recon_high = recon_high
        self.patch_size = patch_size

        # Calculate number of patches along each dimension
        with h5py.File(self.recon_low, "r") as p_low:
            depth, height, width = p_low['data'].shape
            self.patches_depth = depth // self.patch_size[0]
            self.patches_height = height // self.patch_size[1]
            self.patches_width = width // self.patch_size[2]

    def __len__(self):
        return self.patches_depth * self.patches_height * self.patches_width

    def __getitem__(self, idx):
        # Calculate patch coordinates based on index
        z = (idx // (self.patches_height * self.patches_width)) * self.patch_size[0]
        y = ((idx % (self.patches_height * self.patches_width)) // self.patches_width) * self.patch_size[1]
        x = (idx % self.patches_width) * self.patch_size[2]
        
        with h5py.File(self.recon_low, "r") as r_low, h5py.File(self.recon_high, "r") as r_high:
            image = r_low['data'][z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]
            label = r_high['data'][z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)

        return image, label
    