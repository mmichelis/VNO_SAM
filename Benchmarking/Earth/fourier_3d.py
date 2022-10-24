"""
@author*: Levi Lingsch
This file will compute predictions on a global scale, using data from NASA's EarthData Data Set. 
We are using the MERRA2_401.tavg1_2d_flx_Nx.***.nc4 data sets from approximately 2019 to 2022. 

*This file was originally authored by Zongyi Li for the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which takes the 2D spatial + 1D temporal equation directly as a 3D problem
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(15, self.width)
        # input channel is 15: the solution of the first 12 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

################################################################
# configs
################################################################

ntrain = 100
ntest = 100

modes = 32
modes_time = 8
width = 40

batch_size = 1
batch_size2 = batch_size

epochs = 50
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

# path = 'test'
DAT = 'QLML'
path = DAT+'_data_'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
# path_model = '../model/'+path
# path_train_err = 'results/'+path+'train.txt'
# path_test_err = 'results/'+path+'test.txt'
# path_image = 'image/'+path


runtime = np.zeros(2, )
t1 = default_timer()


sub = 3
# S = 64 // sub
T_in = 12
T = 12
T_ = 12

################################################################
# load data
################################################################
import pdb


TEST_PATH = f'../../../VNO_data/EarthData/{DAT}_data_0.mat'
reader = MatReader(TEST_PATH)
test_a = reader.read_field(DAT)[-ntest:,:T_in,::sub,::sub]
test_u = reader.read_field(DAT)[-ntest:,T_in:T+T_in,::sub,::sub]

TRAIN_PATH = f'../../../VNO_data/EarthData/{DAT}_data_1.mat'
reader = MatReader(TRAIN_PATH)
train_a = reader.read_field(DAT)[:ntrain,:T_in,::sub,::sub]
train_u = reader.read_field(DAT)[:ntrain,T_in:T+T_in,::sub,::sub]

for NUM in range(2, 5):
    TRAIN_PATH = f'../../../VNO_data/EarthData/{DAT}_data_{NUM}.mat'
    reader = MatReader(TRAIN_PATH)
    train_a = torch.cat((train_a, reader.read_field(DAT)[:ntrain,:T_in,::sub,::sub]))
    train_u = torch.cat((train_u, reader.read_field(DAT)[:ntrain,T_in:T+T_in,::sub,::sub]))



print(train_u.shape)
print(test_u.shape)

# can't assert without knowing shapes beforehand, so I just gather them from the data and use them where necessary
S_x = train_u.shape[-1]
S_y = train_u.shape[-2]

assert (T == train_u.shape[1])

a_normalizer = UnitGaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

# reshape the data to be in [Number of exmaples, X-coordinates, Y-coordinates, 1, Time], this is how their code was originally written
train_a = train_a.reshape(ntrain,S_x,S_y,1,T_in).repeat([1,1,1,T,1])
test_a = test_a.reshape(ntest,S_x,S_y,1,T_in).repeat([1,1,1,T,1])

train_u = train_u.reshape(ntrain,S_x,S_y,T)
test_u = test_u.reshape(ntrain,S_x,S_y,T)

# normalizer must come after reshape
y_normalizer = UnitGaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

################################################################
# training and evaluation
################################################################
train_model = True

training_history = open(f'./training_history/{DAT}_data.txt', 'w')
training_history.write('Epoch  Time  MSE Train_L2 Test_L2 \n')

model = FNO3d(modes, modes, modes_time, width).cuda()
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

train_loss = np.zeros(epochs)
myloss = LpLoss(size_average=False)
y_normalizer.cuda()

# pdb.set_trace()

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x).view(batch_size, S_x, S_y, T)
        # mse.backward()
        # pull only T_ number time steps
        y = y_normalizer.decode(y)[:,:,:,:T_]
        out = y_normalizer.decode(out)[:,:,:,:T_]

        mse = F.mse_loss(out, y, reduction='mean')
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()
        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()
    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x).view(batch_size, S_x, S_y, T)
            out = y_normalizer.decode(out)[:,:,:,:T_]
            y = y[:,:,:,:T_]
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest
    train_loss[ep] = train_l2
    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)
    training_history.write(f'{ep} {t2-t1} {train_mse} {train_l2} {test_l2} \n')
training_history.close()


pred = torch.zeros(test_u.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

prediction_history = open(f'./training_history/{DAT}_data_test_loss.txt', 'w')

with torch.no_grad():
    t1 = default_timer()
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        # ll: had to add '.view(1, S, S, T)'... this matches what is above. Hopefully it prevents the size mismatch error I am getting for the decoding.
        out = model(x).view(1, S_x, S_y, T)
        out = y_normalizer.decode(out)
        pred[index] = out

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        print(index, test_l2)
        index = index + 1
        prediction_history.write(f'{test_l2} \n')
prediction_history.close()



scipy.io.savemat('./predictions/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})