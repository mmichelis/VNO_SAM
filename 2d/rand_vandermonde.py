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
sys.path.append('../')
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

        self.Vx, self.Vx_ct, self.Vy, self.Vy_ct = self.internal_vandermonde()

    def internal_vandermonde(self):
        
        V_x = torch.zeros([self.modes1, S_x], dtype=torch.cfloat).cuda()
        for row in range(self.modes1):
             for col in range(S_x):
                V_x[row, col] = np.exp(-1j * row *  pos_x[0, col]) 
        V_x = torch.divide(V_x, np.sqrt(S_x))

        V_y = torch.zeros([self.modes1, S_y], dtype=torch.cfloat).cuda()
        for row in range(self.modes1):
             for col in range(S_y):
                V_y[row, col] = np.exp(-1j * row *  pos_y[0, col]) 
        V_y = torch.divide(V_y, np.sqrt(S_y))


        return torch.transpose(V_x, 0, 1), torch.conj(V_x), torch.transpose(V_y, 0, 1), torch.conj(V_y)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        # x_ft = torch.fft.rfft2(x)
        # pdb.set_trace()
        x_ft = torch.matmul(
                    torch.transpose(
                        torch.matmul(x.cfloat(), self.Vx)
                    , 2, 3)
                , self.Vy)

        # Multiply relevant Fourier modes
        # out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        # out_ft[:, :, :self.modes1, :self.modes2] = \
        #     self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # out_ft[:, :, -self.modes1:, :self.modes2] = \
        #     self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft = torch.zeros(batchsize, self.out_channels,  self.modes1, self.modes2, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft, self.weights1)
        # out_ft

        #Return to physical space
        # x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        x = torch.matmul(
                torch.transpose(
                    torch.matmul(
                        torch.transpose(out_ft, 2, 3),
                    self.Vx_ct),
                2, 3),
            self.Vy_ct).real
        x = torch.transpose(x, 2, 3)
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
        self.fc0 = nn.Linear(T+2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

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

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[1]
        # gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = pos_x
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_y, 1, 1])
        # gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = pos_y
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################

TRAIN_PATH = '../../VNO_data/rand_ns_V1e-3_N5000_T50.mat'
TEST_PATH = '../../VNO_data/rand_ns_V1e-3_N5000_T50.mat'

ntrain = 1000
ntest = 100

modes = 10
width = 20

batch_size = 20
batch_size2 = batch_size

epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

path = 'rand_ns_fourier_2d_rnn_V10000_T20_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = './VNO_models/'+path
# path_train_err = 'results/'+path+'train.txt'
# path_test_err = 'results/'+path+'test.txt'
# path_image = 'image/'+path

sub = 1
T_in = 25
T = 25
step = 1

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

pos_x = reader.read_field('loc_x')
pos_x = pos_x / pos_x[0,-1] * np.pi * 2
pos_y = reader.read_field('loc_y')
pos_y  = pos_y / pos_y[0,-1] * np.pi * 2

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)
S_x = train_u.shape[-2]
S_y = train_u.shape[-3]
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S_y,S_x,T_in)
test_a = test_a.reshape(ntest,S_y,S_x,T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

# ll: adding y_normalizer to fix size discrepency
# y_normalizer = UnitGaussianNormalizer(test_u)

################################################################
# training and evaluation
################################################################

model = FNO2d(modes, modes, width).cuda()

print(count_params(model))
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
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
            # pdb.set_trace()
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
            # print(loss.item(), test_l2_full, ntest)
    t2 = default_timer()
    scheduler.step()
    # print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
    #       test_l2_full / ntest)
    print(f'epoch: {ep}, train loss: {train_l2_full / ntrain}, test loss: {test_l2_full / ntest}')
torch.save(model, path_model)




# pred = torch.zeros(test_u.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# ll: adding this to put y_norm on cuda
# y_normalizer.cuda()
with torch.no_grad():
    for xx, yy in test_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        
        full_pred = model(xx)
        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        loss = myloss(pred.reshape(1, -1), yy.reshape(1, -1)).item()

        print(index, loss)
        index = index + 1
        full_pred = torch.cat((full_pred, pred), -1)

# ll: save as .txt instead of .mat
scipy.io.savemat('../VNO_predictions/'+path+'.mat', mdict={'pred': full_pred.cpu().numpy()})

