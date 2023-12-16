import torch
import torch.nn as nn
import numpy as np

def ScalingBlock(num_channels):
    """Make a Module that normalizes the input data.

    This part of the network can be used to renormalize the input
    data. Its parameters are

    * saved when the network is saved;
    * not updated by the gradient descent solvers.

    :param num_channels: The number of channels.
    :param conv3d: Indicates that the input data is 3D instead of 2D.
    :returns: A scaling module.
    :rtype: torch.nn.ConvNd

    """
    c = nn.Conv3d(num_channels, num_channels, 1)
    c.bias.requires_grad = False
    c.weight.requires_grad = False

    scaling_module_set_scale(c, 1.0)
    scaling_module_set_bias(c, 0.0)

    return c

def scaling_module_set_scale(sm, s):
    device = sm.weight.data.device
    c_out, c_in = sm.weight.shape[:2]
    assert c_out == c_in
    if isinstance(s, float) or (torch.is_tensor(s) and s.shape[0]==1):
        sm.weight.data.zero_()
        for i in range(c_out):
            sm.weight.data[i, i] = s
    if torch.is_tensor(s) and s.shape[0]>1:
        assert c_out == c_in == s.shape[0]
        sm.weight.data.zero_()
        for i in range(c_out):
            sm.weight.data[i, i] = s[i]
    sm.to(device)


def scaling_module_set_bias(sm, bias):
    device = sm.bias.data.device
    if isinstance(bias, float) :
        sm.bias.data.fill_(bias)
    if torch.is_tensor(bias):
        sm.bias.data = bias
    sm.to(device)

def set_normalization3d(model, dataloader):
    """Normalize input and target data.

    This function goes through all the training data to compute
    the mean and std of the training data.

    It modifies the network so that all future invocations of the
    network first normalize input data and target data to have
    mean zero and a standard deviation of one.

    These modified parameters are not updated after this step and
    are stored in the network, so that they are not lost when the
    network is saved to and loaded from disk.

    Normalizing in this way makes training more stable.

    :param dataloader: The dataloader associated to the training data.
    :returns:
    :rtype:

    """
    print("Calculating the normalization factors")
    mean_in = square_in = mean_out = square_out = 0

    for (data_in, data_out) in dataloader:
        mean_in += data_in.mean(axis=(0,2,3,4))
        mean_out += data_out.mean(axis=(0,2,3,4))
        square_in += data_in.pow(2).mean(axis=(0,2,3,4))
        square_out += data_out.pow(2).mean(axis=(0,2,3,4))

    mean_in /= len(dataloader)
    mean_out /= len(dataloader)
    square_in /= len(dataloader)
    square_out /= len(dataloader)

    std_in = np.sqrt(square_in - mean_in ** 2)
    std_out = np.sqrt(square_out - mean_out ** 2)

    # The input data should be roughly normally distributed after
    # passing through scale_in. Note that the input is first
    # scaled and then recentered.
    scaling_module_set_scale(model.scale_in, 1 / std_in)
    scaling_module_set_bias(model.scale_in, -mean_in / std_in)
    # The scale_out layer should rather 'denormalize' the network
    # output.
    scaling_module_set_scale(model.scale_out, std_out)
    scaling_module_set_bias(model.scale_out, mean_out)