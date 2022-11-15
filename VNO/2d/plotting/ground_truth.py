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

# import the test data
path_test = '../../../../VNO_data/2d/'
training_test =  'navierstokes_512_512_v1e-4_0.mat'
DATA_PATH_TEST = path_test + training_test

# read the ground_truth data
test_reader = MatReader(DATA_PATH_TEST)
data = test_reader.read_field('vorticity')[32,...].numpy()
x, y = np.mgrid[0:512,0:512]

# plot the prediction data
s = 20

# make an animation from the contourf plots as they evolve in time
# define the update and init functions for the animation, call FuncAnimation and save
def update(frame):
    frame = int(frame)
    fig.clf()
    print(frame)
    u = data[:,:,frame]
    cont = plt.contourf(x, y, u, 60, cmap='RdYlBu', marker='.')
    plt.title('t=%i:' % frame)
    return cont
def init():
    u = data[:,:,0]
    cont = plt.contourf(x, y, u,  60, cmap='RdYlBu', marker='.')
    return cont

fig = plt.figure()
ani = animation.FuncAnimation(fig, update, frames=s, interval = 100, init_func=init)
ani.save('ground_truth.mp4', writer=animation.FFMpegWriter())
