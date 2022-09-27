import sys
import scipy.io
import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from matplotlib import cm
import numpy as np

sys.path.append('../../../')

from utilities3 import *

# import the training data
print(f'Loading data.')
dataloader = MatReader('../../../../VNO_data/1d/burgers_data_R10.mat')
x_data = dataloader.read_field('a')[:,:]
y_data = dataloader.read_field('u')[:,:]

# come up with some size for creating new data
num_samples = int(x_data.shape[0])
ar_len = 64

# generate positions randomly
pos = torch.randperm(8192, [ar_len,])

import pdb
pdb.set_trace()

# create the nonuniform data using the positions
x_nu = torch.index_select(x_data, 1, pos)
y_nu = torch.index_select(y_data, 1, pos)

# plot full data with new data
s = x_data.shape[-1]
x = np.linspace(0,s-1,s)


# generate an example plot
fig, ax = plt.subplots(2)
ax[1].scatter(pos, x_nu[1, :], marker='.')
ax[0].scatter(x, x_data[1, :], marker='.')
ax[0].title.set_text('Uniform Points')
ax[1].title.set_text('Uniformly Random Points')
plt.show()

# save the data
scipy.io.savemat('../../../nu_data/rand_burgers_data_R10.mat', mdict={'loc': pos.numpy(), 'a': x_nu.numpy(), 'u': y_nu.numpy()})
