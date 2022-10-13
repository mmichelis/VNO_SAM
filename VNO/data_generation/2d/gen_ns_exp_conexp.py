import sys
from tkinter import N
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
import numpy as np
import torch
import pdb


sys.path.append('../../../')

from utilities3 import *


# 512 points across, so working with two 32 point regions both above and below center
growth_x = 1.6
growth_y = 1.5

# num_samples = int(x_train.shape[0])
ar_len_1 = 512//2 # above and below
ar_len_2 = 512 # to the right

# the new nonuniform length
nu_len_x = int(ar_len_1**(1/growth_x))
print(f'Number of points along x: {nu_len_x}')
nu_len_y = int(np.round(ar_len_2**(1/growth_y)))
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

# expanding section must be offset
exp_x  = exp_x + exp_x[-1] + 1

# concatenate the contracting and expanding sections
pos_x = torch.cat([con_x, exp_x])
pos_y = exp_y

grid = torch.meshgrid(pos_x, pos_y)
pdb.set_trace()
plt.scatter(grid[0], grid[1], 10*np.ones_like(grid[0]))
plt.show()

# import the training data
print(f'Loading data.')
train_dataloader = MatReader('../../../VNO_data/2d/ns_V1e-3_N5000_T50.mat')
x_train = train_dataloader.read_field('u')[:,:,:,:]
print(x_train.shape)

# select indices for nonuniform data
print('Creating nonuniform data.')
x_train = torch.index_select(torch.index_select(x_train, 2, pos_x), 1, pos_y)

# plot for visual
x, y = np.meshgrid(pos_x, pos_y)
u = x_train[0,:,:,0]
cont = plt.contourf(x, y, u,  60, cmap='RdYlBu', marker='.')
cont = plt.scatter(x, y, marker='.', color='k')
plt.show()

print('Saving nonuniform data.')
scipy.io.savemat('../../../VNO_data/exp_conexp_ns_V1e-3_N5000_T50.mat', mdict={'loc_x': pos_x.numpy(), 'loc_y':pos_y.numpy(), 'u': x_train.numpy()})