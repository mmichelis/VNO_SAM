import numpy as np
import torch
import sys
import matplotlib.pyplot as plt

sys.path.append('../../')
from utilities3 import *

reader = MatReader('./predictions/SPEED_data_100_ep5_m32_w40.mat')
prediction = reader.read_field('pred')

lon = prediction.shape[1]
lat = prediction.shape[2]

plt.contourf(np.meshgrid(lon,lat), prediction[0,:,:,0], cmap=bone)
plt.show()