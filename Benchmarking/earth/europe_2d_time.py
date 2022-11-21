"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import sys


import operator
from functools import reduce
from functools import partial

from timeit import default_timer

sys.path.append('../../')
from utilities3 import *
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

import pdb

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

    def forward(self, x):
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

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(14, self.width)
        # input channel is 14: the solution of the previous 12 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

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

        # x2 = self.w0(x)
        # x = F.gelu(x2)
        # x2 = self.w1(x)
        # x = F.gelu(x2)
        # x2 = self.w2(x)
        # x = F.gelu(x2)
        # x2 = self.w3(x)
        # x = F.gelu(x2)

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################

modes = 16
width = 20

batch_size = 1
batch_size2 = batch_size

epochs = 100
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

DAT = 'QLML'
path = DAT+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
t1 = default_timer()

sub = 1
T_in = 12
T = 12
step = 1

center_lon = int(188 * 1.6)
center_lat = 137 * 2
offset = 30
left = center_lon - offset
right = center_lon + offset
bottom = center_lat - offset
top = center_lat + offset
################################################################
# load data
################################################################
def load_data():
    TEST_PATH = f'../../../VNO_data/EarthData/{DAT}_data_0.mat'
    reader = MatReader(TEST_PATH)
    test_a = reader.read_field(DAT)[:,:T_in,bottom:top, left:right]
    test_u = reader.read_field(DAT)[:,T_in:T+T_in,bottom:top, left:right]

    TRAIN_PATH = f'../../../VNO_data/EarthData/{DAT}_data_1.mat'
    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field(DAT)[:,:T_in,bottom:top, left:right]
    train_u = reader.read_field(DAT)[:,T_in:T+T_in,bottom:top, left:right]

    for NUM in range(2, 5):
        TRAIN_PATH = f'../../../VNO_data/EarthData/{DAT}_data_{NUM}.mat'
        reader = MatReader(TRAIN_PATH)
        train_a = torch.cat((train_a, reader.read_field(DAT)[:,:T_in,bottom:top, left:right]))
        train_u = torch.cat((train_u, reader.read_field(DAT)[:,T_in:T+T_in,bottom:top, left:right]))

    return test_a, test_u, train_a, train_u
test_a, test_u, train_a, train_u = load_data()

# I am concatenating several large data file together here, so the ntrain is variable. Should just reset it here with the actual value.
ntrain = train_a.shape[0]
ntest = test_a.shape[0]

print(train_u.shape)
print(test_u.shape)

# can't assert without knowing shapes beforehand, so I just gather them from the data and use them where necessary
S_x = train_u.shape[-1]
S_y = train_u.shape[-2]
lon = np.arange(S_x)
lat = np.arange(S_y)

lon_, lat_ = np.meshgrid(lat, lon)

assert (T == train_u.shape[1])

# NOTE: using reshape will severely alter the data. Use torch. swapaxes instead to get it into the correct format.
# go from (0, 1, 2, 3) to (0, 3, 2, 1)
# train_a = train_a.reshape(ntrain,S_x,S_y,T_in)
# test_a = test_a.reshape(ntest,S_x,S_y,T_in)
train_a = torch.swapaxes(train_a, 1, 3)
test_a = torch.swapaxes(test_a, 1, 3)
train_u = torch.swapaxes(train_u, 1, 3)
test_u = torch.swapaxes(test_u, 1, 3)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')
# pdb.set_trace()
# plt.contourf(lat_, lon_, train_a[100,:,:,0].cpu().numpy(), cmap='RdYlBu')
# plt.show()
################################################################
# training and evaluation
################################################################

model = FNO2d(modes, modes, width).cuda()
# model = torch.load(path_model)


print(count_params(model))
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)

training_history = open(f'./training_history/2d_europe_{DAT}_data.txt', 'w')
training_history.write('Epoch  Time  Train_L2_step Train_L2_full Test_L2_step Test_L2_full \n')

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]

            im = model(xx)

            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)


            for t in range(0, T, step):
                y = yy[..., t:t + step]

                im = model(xx)
                
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    
    print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
          test_l2_full / ntest)
    training_history.write(str(ep)+' '+ str(t2-t1)+' '+ str(train_l2_step / ntrain / (T / step))\
                    +' '+ str(train_l2_full / ntrain)\
                    +' '+ str(test_l2_step / ntest / (T / step))\
                    +' '+ str(test_l2_full / ntest)\
                    +'\n')

training_history.close()
# torch.save(model, path_model)




# pred = torch.zeros(test_u.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
prediction_history = open(f'./training_history/2d_europe_{DAT}_data_test_loss.txt', 'w')
batch_size=1        # need to set this otherwise the loss outputs are not correct
with torch.no_grad():
    for xx, yy in test_loader:
        step_loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        
        for t in range(0, T, step):
            y = yy[..., t:t + step]

            im = model(xx)

            step_loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        full_loss = myloss(pred.reshape(1, -1), yy.reshape(1, -1))
        
        print(index, full_loss.item(), step_loss.item() / T)
        index = index + 1

        prediction_history.write(f'{full_loss.item()}   {step_loss.item() / T}')
prediction_history.close()
print(pred.shape)

scipy.io.savemat('./predictions/2d_europe_'+path+'.mat', mdict={'pred': pred.cpu().numpy()})