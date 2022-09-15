import sys
import scipy.io
import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from matplotlib import cm
import numpy as np

sys.path.append('../../')

from utilities3 import *

# import the training data
print(f'Loading data.')
dataloader = MatReader('../../../data/burgers_data_R10.mat')
x_data = dataloader.read_field('a')[:,:]
y_data = dataloader.read_field('u')[:,:]

# come up with some size for creating new data
num_samples = int(x_data.shape[0])
ar_len = 8192//2
growth_factor = 2.4

# the new nonuniform length
nu_len = int(ar_len**(1/growth_factor))
print(f'Number of points: {nu_len}')

# generate positions for expanding and contracting section
exp = torch.zeros(nu_len)
for index in range(nu_len):
    exp[index] = index ** growth_factor
exp_floor = exp.int()

# contracting section has opposite distribution
con = exp_floor[-1] - exp_floor
con = torch.flip(con, [0])

# expanding section must be offset
exp  = exp_floor + exp_floor[-1] + 1

# concatenate the contracting and expanding sections
pos = torch.cat([con, exp])

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
ax[1].title.set_text('Contracting and Expanding Points')
plt.show()

# save the data
scipy.io.savemat('../../../nu_data/conexp_burgers_data_R10.mat', mdict={'loc': pos.numpy(), 'a': x_nu.numpy(), 'u': y_nu.numpy()})
