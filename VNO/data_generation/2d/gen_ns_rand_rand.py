import sys
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
import numpy as np
import torch
import pdb


sys.path.append('../../../')

from utilities3 import *


# generate positions randomly
ar_len = 64
pos_x = torch.randperm(512)[:ar_len]
pos_y = torch.randperm(512)[:ar_len]


grid = torch.meshgrid(pos_x, pos_y)
pdb.set_trace()
plt.scatter(grid[0], grid[1], 10*np.ones_like(grid[0]))
plt.show()

# import the training data
print(f'Loading data.')
train_dataloader = MatReader('../../../../VNO_data/2d/ns_V1e-3_N5000_T50.mat')
x_train = train_dataloader.read_field('u')[:,:,:,:]
print(x_train.shape)

# select indices for nonuniform data
print('Creating nonuniform data.')
x_train = torch.index_select(torch.index_select(x_train, 2, pos_x), 1, pos_y)


print('Saving nonuniform data.')
scipy.io.savemat('../../../../VNO_data/2d/rand_ns_V1e-3_N5000_T50.mat', mdict={'loc_x': pos_x.numpy(), 'loc_y':pos_y.numpy(), 'u': x_train.numpy()})

# plot for visual
pos_x = np.sort(pos_x)
pos_y = np.sort(pos_y)
x, y = np.meshgrid(pos_x, pos_y)
u = x_train[0,:,:,0]
cont = plt.contourf(x, y, u,  60, cmap='RdYlBu', marker='.')
cont = plt.scatter(x, y, marker='.', color='k')
plt.show()