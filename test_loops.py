import os
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from tifffile import imread, imsave
import h5py

def test_loop_tif(dataloader, model, loss, output_folder, device='cuda'):
    batches = len(dataloader)
    Path(output_folder).mkdir(exist_ok=True)
    bar = tqdm(dataloader)
    test_loss = 0
    i = 0
    with torch.no_grad():
        for X, y in bar:
            X = X.to(device, dtype=torch.float)
            pred = model(X)
            out_path = os.path.join(output_folder, f"output_{i:04d}.tif")
            imsave(out_path, pred.cpu().numpy())
            label = y.to(device, dtype=torch.float)
            cur_loss = loss(pred, label)
            test_loss += cur_loss / batches
            bar.set_description(f"test loss: {cur_loss:>7f}")
            i+=1
        print(f"Avg loss on whole image: {test_loss:>8f} \n")

def test_loop_tif_patch(dataloader, model, loss, output_folder, device='cuda'):
    # Assuming the dataloader is based on the DatasetUnet3D
    patch_size = dataloader.dataset.patch_size
    batches = len(dataloader)

    Path(output_folder).mkdir(exist_ok=True)
    # Create a placeholder HDF5 for reconstructed volume
    placeholder_h5 = os.path.join(output_folder, "reconstructed_volume.h5")
    with h5py.File(placeholder_h5, "w") as f:
        dset = f.create_dataset("reconstructed", shape=(2000, 1250, 1250), dtype=np.float32)

    bar = tqdm(dataloader)
    test_loss = 0
    with torch.no_grad():
        for idx, (X, y) in enumerate(bar):
            X = X.to(device, dtype=torch.float)
            pred = model(X)
            
            z_idx = (idx // (dataloader.dataset.patches_height * dataloader.dataset.patches_width)) * patch_size[0]
            y_idx = ((idx % (dataloader.dataset.patches_height * dataloader.dataset.patches_width)) // dataloader.dataset.patches_width) * patch_size[1]
            x_idx = (idx % dataloader.dataset.patches_width) * patch_size[2]

            # Save the prediction back to the placeholder HDF5
            with h5py.File(placeholder_h5, "a") as f:
                f["reconstructed"][z_idx:z_idx+patch_size[0], y_idx:y_idx+patch_size[1], x_idx:x_idx+patch_size[2]] = pred.cpu().numpy()

            label = y.to(device, dtype=torch.float)
            cur_loss = loss(pred, label)
            test_loss += cur_loss / batches
            bar.set_description(f"test loss: {cur_loss:>7f}")

        print(f"Avg loss on whole image: {test_loss:>8f} \n")

    # After all predictions are saved, read slices from the HDF5 file and save as TIFs
    with h5py.File(placeholder_h5, "r") as f:
        for idx in range(f["reconstructed"].shape[0]):
            slice_data = f["reconstructed"][idx, 15:-15, 15:-15]  # assuming you still want to crop the borders
            out_path = os.path.join(output_folder, f"output_{idx:05d}.tif")
            imsave(out_path, slice_data)

    # Optionally delete the placeholder HDF5 file
    os.remove(placeholder_h5)