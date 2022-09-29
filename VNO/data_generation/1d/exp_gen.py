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


# first half of the data should have regular spacing
# second half of the data should have irregular spacing, growth rate of 1.1
num_samples = int(x_data.shape[0])
ar_len = int(x_data.shape[1])
growth_factor = 2.16

uniform = int(0) # int(ar_len/4)
nonuniform =  int(ar_len**(1/growth_factor)) #int(np.floor(np.log(ar_len - uniform)/np.log(1.1)))+1
nu_len = uniform+nonuniform
print(f'Number of points: {nu_len}')

# generate positional index
pos = torch.zeros(uniform + nonuniform)
zs = torch.zeros(uniform + nonuniform)
for index in range(uniform + nonuniform):
    if(index <= uniform):
        pos[index] = index
    else:
        # pos[index] = int(index + 1.1**(index - uniform) - 1)
        pos[index] = index ** growth_factor

pos_floor = pos.int()

# create the nonuniform data using the positions
x_nu = torch.index_select(x_data, 1, pos_floor)
y_nu = torch.index_select(y_data, 1, pos_floor)

# plot full data with new data
s = x_data.shape[-1]
x = np.linspace(0,s-1,s)

# generate an example plot
fig, ax = plt.subplots(2)
ax[1].scatter(pos, x_nu[1, :], marker='.')
ax[0].scatter(x, x_data[1, :], marker='.')
ax[0].title.set_text('Uniform Points')
ax[1].title.set_text('Expanding Points')
# plt.show()

# save the data
scipy.io.savemat('../../../../VNO_data/1d/vno_exp_burgers_data_R10.mat', mdict={'loc': pos.numpy(), 'a': x_nu.numpy(), 'u': y_nu.numpy()})
