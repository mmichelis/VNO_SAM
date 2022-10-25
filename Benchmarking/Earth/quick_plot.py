import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

sys.path.append('../../')
from utilities3 import *
file = input('file name: ')
reader = MatReader(f'./predictions/{file}')
prediction = reader.read_field('pred')
print(prediction.shape)
lon = np.arange(prediction.shape[1])
lat = np.arange(prediction.shape[2])

lon_, lat_ = np.meshgrid(lat, lon)

plt.contourf(lon_, lat_, prediction[0,:,:,0], 60, cmap='RdYlBu')
plt.show()




fig = plt.figure()
cont = plt.contourf(lat_, lon_, prediction[0,:,:],  60, cmap='RdYlBu')
ax = plt.axes(xlim=(-180, 180), ylim=(-90, 90))

def update(frame_num):
    frame_num = int(frame_num)
    fig.clf()
    u = prediction[frame_num,:,:]
    cont = plt.contourf(lat_, lon_, u, 60, cmap='RdYlBu')
    plt.title('t=%i:' % frame_num)
    return cont
    
def init():
    ax = plt.axes(xlim=(-180,180), ylim=(-90, 90))
    return cont

ani = animation.FuncAnimation(fig, update, frames=np.arange(12), interval = 100, init_func=init)
ani.save('animation.gif', writer=animation.FFMpegWriter())

# reader = MatReader('../../../VNO_data/EarthData/SPEED_data_0.mat')
# prediction = reader.read_field('SPEED')[:2,:2,::3,::3]

# lon = np.arange(prediction.shape[3])
# lat = np.arange(prediction.shape[2])

# lon_, lat_ = np.meshgrid(lon, lat)

# plt.contourf(lon_, lat_, prediction[0,0,:,:], 60, cmap='RdYlBu')
# plt.show()