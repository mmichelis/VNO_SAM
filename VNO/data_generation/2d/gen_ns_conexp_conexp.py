import sys
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
import numpy as np
import torch
import pdb

sys.path.append('../../')

from utilities3 import *

# # import the training data
print(f'Loading data.')
train_dataloader = MatReader('../../../data/ns_V1e-3_N5000_T50.mat')
x_train = train_dataloader.read_field('u')[:,:,:,:]
print(x_train.shape)


# 64 points across, so working with two 32 point regions both above and below center
growth_x = 1.5
growth_y = 1.2

# num_samples = int(x_train.shape[0])
ar_len = 32

# the new nonuniform length
nu_len_x = int(ar_len**(1/growth_x))
print(f'Number of points along x: {nu_len_x}')
nu_len_y = int(ar_len**(1/growth_y))
print(f'Number of points along y: {nu_len_y}')

# generate positions for expanding and contracting section
exp_x = torch.zeros(nu_len_x)
exp_y = torch.zeros(nu_len_y)
for index in range(nu_len_x):
    exp_x[index] = index ** growth_x
exp_x = exp_x.int()
for index in range(nu_len_y):
    exp_y[index] = index ** growth_y
exp_y = exp_y.int()

# contracting section has opposite distribution
con_x = exp_x[-1] - exp_x
con_x = torch.flip(con_x, [0])
con_y = exp_y[-1] - exp_y
con_y = torch.flip(con_y, [0])

# expanding section must be offset
exp_x  = exp_x + exp_x[-1] + 1
exp_y  = exp_y + exp_y[-1] + 1

# concatenate the contracting and expanding sections
pos_x = torch.cat([con_x, exp_x])
pos_y = torch.cat([con_y, exp_y])


# select indices for nonuniform data
print('Creating nonuniform data.')
x_train_conexp = torch.index_select(torch.index_select(x_train, 2, pos_x), 1, pos_y)


print('Saving nonuniform data.')
scipy.io.savemat('../../../VNO_data/conexp_ns_V1e-3_N5000_T50.mat', mdict={'loc_x': pos_x.numpy(), 'loc_y':pos_y.numpy(), 'u': x_train_conexp.numpy()})