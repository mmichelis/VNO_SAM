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
dataloader = MatReader('../../../nu_data/rand_burgers_data_R10.mat')
x_data = dataloader.read_field('a')[-100:,:]
y_data = dataloader.read_field('u')[-100:,:]
loc_data = dataloader.read_field('loc')[:,:]
print(x_data.shape)
print(y_data.shape)
print(loc_data.shape)

ex = 4
predloader = MatReader('../VNO_predictions/rand_burger_test.mat')
y_pred = predloader.read_field('pred')[:,:]
print(y_pred.shape)

# sort the data before plotting
loc_sort, x_sort = zip(*sorted(zip(loc_data[0,:].numpy(), x_data[ex,:].numpy())))
loc_sort, y_sort = zip(*sorted(zip(loc_data[0,:].numpy(), y_data[0,:].numpy())))
loc_sort, p_sort = zip(*sorted(zip(loc_data[0,:].numpy(), y_pred[0,:].numpy())))

# plot ground truth against the prediction
fig, ax = plt.subplots()
ax.plot(loc_sort, x_sort, label='Initial State', marker='o')
ax.plot(loc_sort, y_sort, label='Ground Truth', marker='o')
ax.plot(loc_sort, p_sort, label='VNO Solution', marker='o')
plt.rcParams['figure.figsize'] = (10,6)
fig.suptitle('VNO Performance on a Uniformly Random Distribution', fontsize=20)
plt.legend(loc='upper left', fontsize=18)
plt.show()
