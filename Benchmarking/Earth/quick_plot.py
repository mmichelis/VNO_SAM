import numpy as np
import torch
import sys
import matplotlib.pyplot as plt

sys.path.append('../../')
from utilities3 import *

reader = MatReader('./predictions/SPEED_data_100_ep5_m32_w40.mat')
prediction = reader.read_field('pred')

lon = np.arange(prediction.shape[1])
lat = np.arange(prediction.shape[2])

lon_, lat_ = np.meshgrid(lat, lon)

plt.contourf(lon_, lat_, prediction[0,:,:,0], 60, cmap='RdYlBu')
plt.show()

# reader = MatReader('../../../VNO_data/EarthData/SPEED_data_0.mat')
# prediction = reader.read_field('SPEED')[:2,:2,::3,::3]

# lon = np.arange(prediction.shape[3])
# lat = np.arange(prediction.shape[2])

# lon_, lat_ = np.meshgrid(lon, lat)

# plt.contourf(lon_, lat_, prediction[0,0,:,:], 60, cmap='RdYlBu')
# plt.show()