import sys
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import interpolate
import torch
import pdb

sys.path.append('../../')

from utilities3 import *

distributions = {'conexp', 'exp', 'rand'}

for dist in distributions:
    pdb.set_trace()

    # import the training data
    print(f'Loading data.')
    dataloader = MatReader('../../../VNO_data/1d/'+dist+'_burgers_data_R10.mat')
    x_data = dataloader.read_field('a')[:,:]
    y_data = dataloader.read_field('u')[:,:]
    loc = dataloader.read_field('loc')[:,:]

    # port everything to numpy
    x_data = x_data.numpy()
    y_data = y_data.numpy()
    sparse_loc = loc.numpy()

    max = np.int(np.amax(sparse_loc) + 1)
    min = np.int(np.amin(sparse_loc))
    d = np.arange(min, max)

    x_dense = np.zeros([2048,max-min])
    y_dense = np.zeros([2048,max-min])

    # the array to hold all the data until ready to port to torch
    for id in range(2048):
        fx = interpolate.interp1d(sparse_loc[0,:], x_data[id,:])
        fy = interpolate.interp1d(sparse_loc[0,:], y_data[id,:])

        x_dense[id,:] = fx(d)
        y_dense[id,:] = fy(d)


    print('Saving '+dist+' data.')
    scipy.io.savemat('../../../VNO_data/1d/full_from_'+dist+'_burgers_data_R10.mat', mdict={'u': x_dense,'a': y_dense})