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


# generate positions randomly
pos_x = torch.IntTensor([4, 41, 24, 50, 37, 28, 35, 11, 20, 7, 30, 34, 17, 26, 45])
pos_y = torch.IntTensor([2, 13, 18, 41, 54, 38, 43, 15, 16, 31, 39, 63, 10, 29, 44])



# select indices for nonuniform data
print('Creating nonuniform data.')
x_train = torch.index_select(torch.index_select(x_train, 2, pos_x), 1, pos_y)


print('Saving nonuniform data.')
scipy.io.savemat('../../../VNO_data/rand_ns_V1e-3_N5000_T50.mat', mdict={'loc_x': pos_x.numpy(), 'loc_y':pos_y.numpy(), 'u': x_train.numpy()})

# plot for visual
pos_x = np.sort(pos_x)
pos_y = np.sort(pos_y)
x, y = np.meshgrid(pos_x, pos_y)
u = x_train[0,:,:,0]
cont = plt.contourf(x, y, u,  60, cmap='RdYlBu', marker='.')
cont = plt.scatter(x, y, marker='.', color='k')
plt.show()