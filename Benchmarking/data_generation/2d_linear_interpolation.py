import sys
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate
import torch
import pdb

sys.path.append('../../')

from utilities3 import *


# import the training data
print(f'Loading data.')
train_dataloader = MatReader('../../../VNO_data/conexp_ns_V1e-3_N5000_T50.mat')
x_train = train_dataloader.read_field('u')[:,:,:,:]
loc_x = train_dataloader.read_field('loc_x')
loc_y = train_dataloader.read_field('loc_y')
print(x_train.shape)

# port everything to numpy
x_train = x_train.numpy()
sparse_x = loc_x.numpy()
sparse_y = loc_y.numpy()

sparse_x, sparse_y = np.meshgrid(sparse_x, sparse_y)

# prepare some flattened tensors for the original (sparse) and new (dense) positions
sparse_loc = np.stack((sparse_x.flatten(), sparse_y.flatten()), axis=1)

size = 56
d = np.arange(size)
dx, dy = np.meshgrid(d, d)
dense_loc = np.stack((dx.flatten(), dy.flatten()), axis=1)

# the array to hold all the data until ready to port to torch
full_dense_data = np.zeros([1100, size, size, 50])

# loop through each tensor
for id in range(1000):
    
    # loop through each time step
    for time in range(50):

        # flatten the data
        sparse_data = x_train[id, :, :, time].flatten()
        dense_data = scipy.interpolate.griddata(sparse_loc, sparse_data, dense_loc)
        dense_data = dense_data.reshape(size,size)
        full_dense_data[id, :, :, time] = dense_data

        # if time%10 == 0: 
        #     plt.contourf(dx, dy, dense_data)
        #     plt.scatter(dx, dy, marker='.')
        #     plt.show()

# loop through each tensor
for id in range(100):
    
    # loop through each time step
    for time in range(50):

        # flatten the data
        sparse_data = x_train[-(100 - id), :, :, time].flatten()
        dense_data = scipy.interpolate.griddata(sparse_loc, sparse_data, dense_loc)
        dense_data = dense_data.reshape(size,size)
        full_dense_data[-(100 - id), :, :, time] = dense_data

pdb.set_trace()
print('Saving uniform data.')
scipy.io.savemat('/userdata/llingsch/SAM/VNO_data/full_from_conexp_ns_V1e-3_N1100_T50.mat', mdict={'u': full_dense_data})