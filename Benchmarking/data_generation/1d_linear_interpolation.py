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

distributions = {'conexp', 'exp', 'rand'}

for dist in distributions:
    pdb.set_trace()

    # import the training data
    print(f'Loading data.')
    dataloader = MatReader('../../../VNO_data/1d/full_from_'+dist+'_burgers_data_R10.mat')
    x_data = dataloader.read_field('a')[:,:]
    y_data = dataloader.read_field('u')[:,:]
    loc = dataloader.read_field('loc')[:,:]

    # port everything to numpy
    x_data = x_data.numpy()
    y_data = y_data.numpy()
    sparse_loc = loc.numpy()

    size = 8193
    d = np.arange(size)

    # the array to hold all the data until ready to port to torch
    # x_dense = np.zeros([x.data.shape[0], size])
    # y_dense = np.zeros([y.data.shape[0], size])
    x_dense = scipy.interpolate.griddata(sparse_loc, x_data, d)
    y_dense = scipy.interpolate.griddata(sparse_loc, y_data, d)



    # loop through each tensor
    # for id in range(x.data.shape[0]):
    #     sparse_data = x_train[id, :].flatten()
    #     dense_data = scipy.interpolate.griddata(sparse_loc, sparse_data, dense_loc)
    #     dense_data = dense_data.reshape(size,size)
    #     full_dense_data[id, :] = dense_data


    print('Saving '+dist+' data.')
    scipy.io.savemat('../../../VNO_data/1d/full_from_'+dist+'_burgers_data_R10.mat', mdict={'u': x_dense,'a': y_dense})