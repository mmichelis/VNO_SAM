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

lat_, lon_ = np.meshgrid(lat, lon)

plt.contourf(lat_, lon_, prediction[0,:,:,0], 60, cmap='RdYlBu_r')
plt.show()


# save an animation

fig = plt.figure()
cont = plt.contourf(lat_, lon_, prediction[0,:,:,0],  60, cmap='RdYlBu_r')
ax = plt.axes(xlim=(-180, 180), ylim=(-90, 90))

def update(frame_num):
    frame_num = int(frame_num)
    fig.clf()
    u = prediction[0,:,:,frame_num]
    cont = plt.contourf(lat_, lon_, u, 60, cmap='RdYlBu_r')
    plt.title('t=%i:' % frame_num)
    return cont
    
def init():
    ax = plt.axes(xlim=(-180,180), ylim=(-90, 90))
    return cont

ani = animation.FuncAnimation(fig, update, frames=12, interval = 100, init_func=init)
ani.save(file[:-4]+'.gif', writer=animation.FFMpegWriter())

