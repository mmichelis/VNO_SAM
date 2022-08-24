import scipy.io
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../')
from utilities3 import *
import numpy as np

# import the training data
dataloader = MatReader('../../../nu_data/exp_burgers_data_R10.mat')
x_data = dataloader.read_field('a')[-100:,:]
y_data = dataloader.read_field('u')[-100:,:]
loc_data = dataloader.read_field('loc')[:,:]
print(x_data.shape)
print(y_data.shape)
print(loc_data.shape)

ex = 1
predloader = MatReader('../VNO_predictions/exp_burger_test.mat')
y_pred = predloader.read_field('pred')[:,:]
print(y_pred.shape)

# plot ground truth against the prediction
fig, ax = plt.subplots()
ax.plot(loc_data[0,:], x_data[ex,:], label='Initial State', marker='o')
ax.plot(loc_data[0,:], y_data[ex,:], label='Ground Truth', marker='o')
ax.plot(loc_data[0,:], y_pred[ex,:], label='VNO Solution', marker='o')
plt.rcParams['figure.figsize'] = (10,6)
fig.suptitle('VNO Performance on an Expanding Distribution', fontsize=20)
plt.legend(loc='upper left', fontsize=18)
plt.show()
