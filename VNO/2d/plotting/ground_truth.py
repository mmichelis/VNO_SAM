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


# read in the prediction
path_pred = '../predictions/'
pred_file = input('file name: ')

DATA_PATH_TRAIN = path_pred + pred_file
pred_reader = MatReader(DATA_PATH_TRAIN)

# create grid
pos_x = pred_reader.read_field('x_pos')[0,:].numpy().astype(int)
pos_y = pred_reader.read_field('y_pos')[0,:].numpy().astype(int)
x, y = np.meshgrid(pos_x, pos_y)

# import the test data
path_test = '../../../../VNO_data/2d/'
training_test =  'navierstokes_512_512_v1e-4_0.mat'
DATA_PATH_TEST = path_test + training_test

# read the ground_truth data
test_reader = MatReader(DATA_PATH_TEST)
data = test_reader.read_field('vorticity')[0,...].numpy()
# data = np.take(np.take(data, pos_x, axis=0), pos_y, axis=1)

# plot the prediction data
s = 21
fig = plt.figure()
u = data[:,:,0]
cont = plt.contourf(x, y, u,  60, cmap='RdYlBu', marker='.')
# cont = plt.scatter(x, y, marker='.', color='k')
# plt.scatter(x, y, marker)
# make an animation from the contourf plots as they evolve in time
# define the update and init functions for the animation, call FuncAnimation and save
def update(frame):
    frame = int(frame)
    fig.clf()
    print(frame)
    u = data[:,:,frame]
    cont = plt.contourf(x, y, u, 60, cmap='RdYlBu', marker='.')
    # cont = plt.scatter(x, y, marker='.', color='k')
    plt.title('t=%i:' % frame)
    return cont
def init():
    ax = plt.axes(xlim=(0, 511), ylim=(0, 511))
    return cont
ani = animation.FuncAnimation(fig, update, frames=s, interval = 100, init_func=init)
ani.save('ground_truth.gif', writer=animation.FFMpegWriter())
