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
import os

# read in the prediction
path_pred = '../predictions/'
print(os.listdir(path_pred))
pred_file = input('file name: ')

DATA_PATH_TRAIN = path_pred + pred_file
pred_reader = MatReader(DATA_PATH_TRAIN)
data = pred_reader.read_field('pred')[:,:,:,:]

pdb.set_trace()

# create grid
pos_x = np.arange(data.shape[1])
pos_y = np.arange(data.shape[2])
x, y = np.meshgrid(pos_x, pos_y)


# plot the prediction data
s = 10
fig = plt.figure()
u = data[0,:,:,0]
cont = plt.contourf(x, y, u,  60, cmap='RdYlBu', marker='.')
# cont = plt.scatter(x, y, marker='.', color='k')
# plt.scatter(x, y, marker)
# make an animation from the contourf plots as they evolve in time
# define the update and init functions for the animation, call FuncAnimation and save
def update(frame):
    frame = int(frame)
    fig.clf()
    print(frame)
    u = data[0,:,:,frame]
    cont = plt.contourf(x, y, u, 60, cmap='RdYlBu', marker='.')
    # cont = plt.scatter(x, y, marker='.', color='k')
    plt.title('t=%i:' % frame)
    return cont
def init():
    ax = plt.axes(xlim=(0, 511), ylim=(0, 511))
    return cont
ani = animation.FuncAnimation(fig, update, frames=s, interval = 100, init_func=init)
ani.save('prediction.gif', writer=animation.FFMpegWriter())
