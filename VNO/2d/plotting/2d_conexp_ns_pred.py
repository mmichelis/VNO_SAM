import scipy.io

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('../../../')
from utilities3 import *

import numpy as np
import pdb
pdb.set_trace()

# read in the prediction
path_pred = '../predictions/'
pred_file = input('file name: ')

DATA_PATH_TRAIN = path_pred + pred_file
pred_reader = MatReader(DATA_PATH_TRAIN)
data = pred_reader.read_field('pred')[:,:,:,:]

# create grid
pos_x = pred_reader.read_field('x_pos')
pos_y = pred_reader.read_field('y_pos')
x, y = np.meshgrid(pos_x, pos_y)

# import the test data
# path_test = '../../../../VNO_data/2d/'
# training_test =  'navierstokes_512_512_v1e-4_0.mat'
# DATA_PATH_TEST = path_test + training_test

# # read using the utilities3 data reader
# test_reader = MatReader(DATA_PATH_TEST)[:, pos_x, pos_y, :]

# plot the prediction data
s = 10
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
    ax = plt.axes(xlim=(0, 511), ylim=(0, 511))
    return cont
ani = animation.FuncAnimation(fig, update, frames=s, interval = 100, init_func=init)
ani.save('ani.gif', writer=animation.FFMpegWriter())
