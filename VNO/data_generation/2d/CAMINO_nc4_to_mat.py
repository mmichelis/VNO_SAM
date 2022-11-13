import numpy as np
import netCDF4 as nc
import scipy.io
import pdb
import matplotlib.pyplot as plt

##########################################################
# For checking on my local machine
##########################################################
# name = 'sample_1_time_1.nc'
# ds = nc.Dataset(name)

# sub = 1
# u = ds['u'][::sub,::sub].data
# v = ds['v'][::sub,::sub].data

# shape = u.shape
# y = np.arange(shape[0])
# x = y
# plt.streamplot(x, y, u, v)
# plt.show()

# x,y = np.mgrid[0:shape[0],0:shape[1]]
# dims = len(shape)
# field = np.stack((u,v))
# plt.contourf(x, y, u, 60, cmap='RdYlBu')
# plt.show()
# plt.contourf(x, y, v, 60, cmap='RdYlBu')
# plt.show()

# partials = tuple(np.gradient(i) for i in field)
# jacobian = np.stack(partials).reshape(*(j := (dims,) * 2), *shape)
# curl_mask = np.triu(np.ones(j, dtype=bool), k=1)
# curl = (jacobian[curl_mask] - jacobian[curl_mask.T]).squeeze()
# plt.matshow(curl.T, 60, cmap='RdYlBu')
# plt.quiver(*field[:,:], angles='xy')
# plt.show()
##########################################################

##########################################################
# For Camino
##########################################################
file_path = '../../../../VNO_data/2d/sample/'
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

    plt.streamplot(x, y, u, v)
    plt.show()
    
scipy.io.savemat(f'../../../../VNO_data/2d/sample/navierstokes_512_512_v1e-4_{0}.mat', mdict={'u':velocity_field})
##########################################################
##########################################################
# file_path = '../../../../VNO_data/2d/sample/'
# num_samples = 1
# num_timesteps = 21
# num_points = 512

# vorticity_field = np.zeros((1, num_points, num_points, num_timesteps))
# pdb.set_trace()


# for time in range(num_timesteps):
#     name = f'sample_0_time_{time}.nc'
#     ds = nc.Dataset(file_path + name)
#     u = ds['u'][:].data
#     v = ds['v'][:].data
#     shape = u.shape
    # x,y = np.mgrid[0:shape[0],0:shape[1]]
    # dims = len(shape)
    # field = np.stack((u,v))
    # partials = tuple(np.gradient(i) for i in field)
    # jacobian = np.stack(partials).reshape(*(j := (dims,) * 2), *shape)
    # curl_mask = np.triu(np.ones(j, dtype=bool), k=1)
    # curl = (jacobian[curl_mask] - jacobian[curl_mask.T]).squeeze()
    # vorticity_field[index, :,:, time] = curl
# scipy.io.savemat(f'../../../../VNO_data/2d/sample/navierstokes_512_512_v1e-4_{0}.mat', mdict={'u':vorticity_field})
##########################################################