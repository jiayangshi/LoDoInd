import os
from pathlib import Path
import socket
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pcfv import train_loop, psnr, set_normalization, plot_images, valid_loop, plot_images, EarlyStopping
from torch.utils.data import DataLoader
from datasets import DatasetStackTIF, DatasetPatch
from skimage.metrics import peak_signal_noise_ratio as psnr
from test_loops import test_loop_tif, test_loop_tif_patch
from utils import set_normalization3d

# Get current host
HOSTNAME = socket.gethostname()

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

mode = '2.5D'
assert mode in ['2D', '2.5D', '3D']

stack = 5
if mode == '2D':
    stack = 1

working_folder = 'denoise_'+ mode
Path(working_folder).mkdir(exist_ok=True)

# model to denoise
from msd_pytorch import MSDRegressionModel
if mode == '3D':
    model = MSDRegressionModel(1, 1, 50, 1, dilations=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ndim=3)
    test_loop = test_loop_tif_patch
    train_folder = '/media/shij3/T7/LoDoInd/Noise1.h5'
    test_folder = train_folder[:-3] + '_test.h5'
    path_parts = os.path.split(train_folder)
    ref_folder = os.path.join(*path_parts[:-1], "Ref.h5")
    ref_test_folder = ref_folder[:-3] + '_test.h5'
else:
    model = MSDRegressionModel(stack, 1, 100, 1, dilations=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ndim=2)
    test_loop = test_loop_tif
    train_folder = '/media/shij3/T7/LoDoInd/Noise1'
    test_folder = train_folder + '_test'
    path_parts = os.path.split(train_folder)
    ref_folder = os.path.join(*path_parts[:-1], "Ref")
    ref_test_folder = ref_folder + '_test'

train_batch_size = 1
workers = 8

epochs = 100

if mode == '3D':
    training_data = DatasetPatch(train_folder, ref_folder, patch_size=(16,1250,1250))
else:
    training_data = DatasetStackTIF(train_folder, ref_folder, stack=stack)
training_data, validate_data = torch.utils.data.random_split(training_data, [(int)(len(training_data)*0.75), len(training_data) - (int)(len(training_data)*0.75)])
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True,num_workers=workers)
validate_dataloader = DataLoader(validate_data, batch_size=1, shuffle=False,num_workers=workers//2)
if mode == '3D':
    test_data = DatasetPatch(test_folder, ref_test_folder, patch_size=(16,1250,1250))
else:
    test_data = DatasetStackTIF(test_folder, ref_test_folder, stack=stack)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False,num_workers=workers//2)

if mode == '3D':
    set_normalization3d(model, train_dataloader)
else:
    set_normalization(model, train_dataloader)

model = model.net

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
mse_loss = torch.nn.MSELoss()

model = model.to(device)

training_losses = []
validate_losses = []
early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join(working_folder,"weights.pt"))

intermediate_folder = os.path.join(working_folder,"intermidate/")
if not Path(intermediate_folder).exists():
    print("Creating folder for intermediate results")
    Path(intermediate_folder).mkdir()
inter_x, inter_y = validate_data[0]
inter_x_cuda = torch.from_numpy(np.expand_dims(inter_x, (0))).float().to(device)
vmin, vmax = inter_y.min(), inter_y.max()


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    if t==0 or (t+1) % 5 == 0:
        with torch.no_grad():
            inter_prey_cuda = model(inter_x_cuda)
        inter_prey = inter_prey_cuda.detach().cpu().numpy()[0]
        if mode == '3D':
             fig = plot_images(inter_x[0,0], inter_prey[0,0], inter_y[0,0],
                          style=plt.gray(), t1="original", t2="intermediate result",
                          t3="ground truth", subposition=(1, 3), vmin=vmin, vmax=vmax, range=vmax - vmin,
                          show_image=False,
                          x1=f"psnr:{psnr(inter_x[0], inter_y[0], data_range=vmax - vmin):>.2f}dB",
                          x2=f"{psnr(inter_prey, inter_y, data_range=vmax - vmin):>.2f}dB",
                          )
        else:
            fig = plot_images(inter_x[0], inter_prey[0], inter_y[0],
                          style=plt.gray(), t1="original", t2="intermediate result",
                          t3="ground truth", subposition=(1, 3), vmin=vmin, vmax=vmax, range=vmax - vmin,
                          show_image=False,
                          x1=f"psnr:{psnr(inter_x[0], inter_y[0], data_range=vmax - vmin):>.2f}dB",
                          x2=f"{psnr(inter_prey, inter_y, data_range=vmax - vmin):>.2f}dB",
                          )
        fig.savefig(os.path.join(intermediate_folder, f"intermediate_epoch_{t + 1}.png"))
        test_loop(test_dataloader, model, mse_loss, os.path.join(working_folder, 'cleaned'), device)

    loss2 = train_loop(train_dataloader, model, optimizer, mse_loss, device)
    training_losses.append(loss2)
    loss3 = valid_loop(validate_dataloader, model, mse_loss, device)
    validate_losses.append(loss3)
    
    early_stopping(loss3, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

model.load_state_dict(torch.load(os.path.join(working_folder,"weights.pt")))

fig = plt.figure(frameon=True)
plt.plot(training_losses, '-')
plt.plot(validate_losses, '-')
plt.xlabel('epoch')
plt.ylabel('mse loss')
plt.legend(['Train', 'Validate'])
plt.title('Train loss')
fig.savefig(os.path.join(working_folder,"train_loss.png"))

test_loop(test_dataloader, model, mse_loss, os.path.join(working_folder, 'cleaned'), device)
