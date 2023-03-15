"""
This work was originally authored by Zongyi Li, to present the Fourier Neural Operator for 2D+time problem such as the Navier Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).

It has been modified by
@author: Levi Lingsch
to implement the 2D+time VNO on the Navier Stokes equation using data with a contracting-expanding distribution in both dimensions.
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


import operator
from functools import reduce
from functools import partial

from timeit import default_timer

import sys
sys.path.append('../../')
from vft import fully_nonequispaced_vft
from vft import vft2d
from Adam import Adam
from utilities3 import *
import pdb

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))


    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def ndft_forward(self, x):
        batchsize = x.shape[0]
        num_pts = x.shape[-1]

        x = torch.reshape(x, (batchsize, self.out_channels, num_pts**2, 1))
        # x [4, 20, 512, 512]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = ndft_transformer.forward(x.cfloat()) #[4, 20, 32, 16]
        x_ft = torch.reshape(x_ft, (batchsize, self.out_channels, self.modes1, self.modes1))

        # # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, self.modes1, self.modes1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes1] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes1], self.weights1)

        # #Return to physical space
        x_ft = torch.reshape(out_ft, (batchsize, self.out_channels, self.modes1**2, 1))
        x = ndft_transformer.inverse(x_ft) # x [4, 20, 512, 512]
        x = torch.reshape(x, (batchsize, self.out_channels, num_pts, num_pts))

        return x.real
    
    def fft_forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
    def vft_forward(self, x):
        batchsize = x.shape[0]
        # x [4, 20, 512, 512]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = vft_transformer.forward(x.cfloat()) #[4, 20, 32, 16]

        # Multiply relevant Fourier modes
        # out_ft = torch.zeros(batchsize, self.out_channels,  2 * self.modes1, self.modes2, dtype=torch.cfloat, device=x.device)
        x_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        x_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = vft_transformer.inverse(x_ft).real # x [4, 20, 512, 512]

        return x

################################################################
# configs
################################################################
# define which (nonequispaced) data to work with
# options are 'conexp_conexp', 'exp_conexp', 'rand_rand'
data_dist = 'uniform'

file_path = '../../../VNO_data/2d/'
# file_path = '/cluster/scratch/llingsch/NS/'

ntest = 64

width = 20
ndft_modes = 16
vft_modes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

batch_size = 1

T_in = 10
T = 10

################################################################
# load data
################################################################
t1 = default_timer()
print('Preprocessing Data...')

def load_data():
    TRAIN_PATH = f'{file_path}navierstokes_512_512_v1e-4_{0}.mat'
    reader = MatReader(TRAIN_PATH)
    test_a = reader.read_field('vorticity')[:batch_size,:,:,:T_in]

    test_a = torch.reshape(test_a, (batch_size, T_in, 512, 512))

    return test_a
test_a = load_data()

def define_positions(size):

    x_pos = torch.arange(size)
    y_pos = torch.arange(size)

    x_grid, y_grid = torch.meshgrid(x_pos, y_pos)

    x_flat = torch.flatten(x_grid)
    y_flat = torch.flatten(y_grid)

    return x_pos, y_pos, x_flat, y_flat


t2 = default_timer()

print(f'Processing finished in {t2-t1} seconds.')

################################################################
# training and evaluation
################################################################

spectral_conv = SpectralConv2d_fast(width, width, ndft_modes, ndft_modes)

training_history = open('./training_history/matmul_experiments.txt', 'w')
training_history.write('Size    Time    Method \n')

sizes = [32, 64, 128, 256, 512]
for size in sizes:
    x_pos, y_pos, x_flat, y_flat = define_positions(size)

    ndft_transformer = fully_nonequispaced_vft(x_flat, y_flat, ndft_modes)
    vft_transformer = vft2d(x_pos, y_pos, vft_modes, vft_modes)

    x = test_a[:,:,size,size]
    t_ndft = SpectralConv2d_fast.ndft_forward(x)

    print(size)



    
