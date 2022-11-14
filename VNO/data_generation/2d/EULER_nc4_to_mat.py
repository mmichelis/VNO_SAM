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
# For Euler
##########################################################
file_path = '/cluster/scratch/llingsch/NS/bm_H50_N512/'
num_samples = 1024
num_timesteps = 21
num_points = 512
subsets = 16
sub_len = num_samples//subsets

vorticity_field = np.zeros((sub_len, num_points, num_points, num_timesteps))
pdb.set_trace()
for s in range(subsets):
    for index, sample in enumerate(range(s*sub_len, (s+1)*sub_len)):
        for time in range(num_timesteps):
            name = f'sample_{sample}_time_{time}.nc'
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
            vorticity_field[index, :,:, time] = curl

    scipy.io.savemat(f'/cluster/scratch/llingsch/NS/navierstokes_512_512_v1e-4_{s}.mat', mdict={'vorticity':vorticity_field})
##########################################################