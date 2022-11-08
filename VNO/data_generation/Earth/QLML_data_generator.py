import numpy as np
import netCDF4 as nc
import scipy.io
import pdb

#pdb.set_trace()

file_names = np.loadtxt('NewNames.txt', dtype=str)
num_files = file_names.shape[0]
file_name = file_names[0]
file_path = '/cluster/scratch/llingsch/'
name = 'QLML'
ds = nc.Dataset(file_path + file_name)
lat_ = ds['lat'][:].data
lon_ = ds['lon'][:].data
speed = ds[name][:,:,:].data

speed_data = np.empty((100,speed.shape[0],speed.shape[1],speed.shape[2]))

for rd in range(2, 5):
    print(round)
    for id in range(100): #num_files):
        file_name = file_names[id+100*rd]
        ds = nc.Dataset(file_path + file_name)

        speed_data[id] = ds[name][:,:,:].data
        #pdb.set_trace()
    scipy.io.savemat(f'/cluster/scratch/llingsch/{name}_data_{rd}.mat', mdict={name:speed_data, 'lat':lat_, 'lon':lon_})
