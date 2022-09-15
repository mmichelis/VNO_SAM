import scipy.io

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('../')
from utilities3 import *

import numpy as np
import pdb
pdb.set_trace()

###
# import the test data
path_test = '../../../VNO_data/'
training_test =  'conexp_ns_V1e-3_N5000_T50.mat'
DATA_PATH_TEST = path_test + training_test


# read using the utilities3 data reader
test_reader = MatReader(DATA_PATH_TEST)

# create grid
pos_x = test_reader.read_field('loc_x')
pos_y = test_reader.read_field('loc_y')
x, y = np.meshgrid(pos_x, pos_y)

# read in the prediction
path_pred = '../VNO_predictions/'
pred_file = 'conexp_ns_fourier_2d_rnn_V10000_T20_N1000_ep500_m12_w20.mat'

DATA_PATH_TRAIN = path_pred + pred_file
pred_reader = MatReader(DATA_PATH_TRAIN)
data = pred_reader.read_field('pred')[:,:,:,:]
print(data.shape)

# plot the prediction data
s = 25
fig = plt.figure()
u = data[0,:,:,0]
cont = plt.contourf(x, y, u,  60, cmap='RdYlBu', marker='.')
cont = plt.scatter(x, y, marker='.', color='k')
# plt.scatter(x, y, marker)
# make an animation from the contourf plots as they evolve in time
# define the update and init functions for the animation, call FuncAnimation and save
def update(frame):
    frame = int(frame)
    fig.clf()
    print(frame)
    u = data[0,:,:,frame]
    cont = plt.contourf(x, y, u, 60, cmap='RdYlBu', marker='.')
    cont = plt.scatter(x, y, marker='.', color='k')
    plt.title('t=%i:' % frame)
    return cont
def init():
    ax = plt.axes(xlim=(0, 63), ylim=(0, 63))
    return cont
ani = animation.FuncAnimation(fig, update, frames=np.arange(s), interval = 100, init_func=init)
ani.save('../../../Report_Images/2d_time_Vandermonde_predictions.mp4', writer=animation.FFMpegWriter())
