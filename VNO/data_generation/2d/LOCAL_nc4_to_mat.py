import numpy as np
import netCDF4 as nc
import scipy.io
import pdb
import matplotlib.pyplot as plt


##########################################################
# For Local Machine
##########################################################
file_path = './sample/'
num_samples = 1
num_timesteps = 21
num_points = 512

velocity_field = np.zeros((2, num_points, num_points, num_timesteps))

y = np.arange(num_points)
x = y


for time in range(num_timesteps):
    name = f'sample_0_time_{time}.nc'
    ds = nc.Dataset(file_path + name)
    u = ds['u'][:].data
    v = ds['v'][:].data
    shape = u.shape
    velocity_field[0,:,:,time] = u
    velocity_field[1,:,:,time] = v

    # plt.streamplot(x, y, u, v)
    # plt.show()

scipy.io.savemat(f'./sample/velocity.mat', mdict={'velocity':velocity_field})
##########################################################
##########################################################
file_path = './sample/'
num_samples = 1
num_timesteps = 21
num_points = 512

vorticity_field = np.zeros((1, num_points, num_points, num_timesteps))


for time in range(num_timesteps):
    name = f'sample_0_time_{time}.nc'
    ds = nc.Dataset(file_path + name)
    u = ds['u'][:].data
    v = ds['v'][:].data
    shape = u.shape
    x,y = np.mgrid[0:shape[0],0:shape[1]]
    dims = len(shape)
    field = np.stack((u,v))
    partials = tuple(np.gradient(i) for i in field)
    jacobian = np.stack(partials).reshape(*(j := (dims,) * 2), *shape)
    curl_mask = np.triu(np.ones(j, dtype=bool), k=1)
    curl = (jacobian[curl_mask] - jacobian[curl_mask.T]).squeeze()
    vorticity_field[0, :,:, time] = curl

    # plt.contourf(x,y,curl, cmap = 'RdYlBu')
    # plt.show()
scipy.io.savemat(f'./sample/vorticity.mat', mdict={'u':vorticity_field})
##########################################################