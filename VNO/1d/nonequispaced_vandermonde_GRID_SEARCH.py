"""
This work was originally authored by Zongyi Li, to present the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).

It has been modified by
@author: Levi Lingsch
to implement the 1D VNO on the time independent Burgers equation using data with a contracting-expanding distribution.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer

import sys
sys.path.append('../../')
from utilities3 import *
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does VFFT, linear transform, and Inverse VFFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.V, self.V_ct = self.internal_vandermonde()


    # Complex multiplication and complex batched multiplications
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def internal_vandermonde(self):
        
        V = torch.zeros([self.modes1, s], dtype=torch.cfloat).cuda()
        for row in range(self.modes1):
             for col in range(s):
                V[row, col] = np.exp(-1j * row *  p_data[0,col,0]) 
        V = torch.divide(V, np.sqrt(s))
        return torch.transpose(V, 0, 1), torch.conj(V)

    def forward(self, x):
        x_ft = torch.matmul(x.cfloat(), self.V)
        # x_ft -= torch.min(x_ft.real)

        # Multiply relevant Fourier modes
        out_ft = self.compl_mul1d(x_ft, self.weights1)

        x = torch.matmul(out_ft, self.V_ct).real

        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)

        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic
        
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

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        # gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = p_data
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

################################################################
#  configurations
################################################################
ntrain = 1000
ntest = 200

batch_size = 20
learning_rate_ = {0.001, 0.0009, 0.0015}

epochs = 500
step_size = 50
gamma = 0.5

modes_ = {32}
width_ = {64}

weight_decay_ = {2e-5, 1e-5, 5e-6}


################################################################
# read data
################################################################
# define which (nonequispaced) data to work with
# options are 'conexp', 'exp', 'rand'
data_dist = input('data distribution: conexp, exp, or rand?\n')

# Data is of the shape (number of samples, grid size)
dataloader = MatReader('../../../VNO_data/1d/vno_'+data_dist+'_burgers_data_R10.mat')
x_data = dataloader.read_field('a')[:,:]
y_data = dataloader.read_field('u')[:,:]
p_data = dataloader.read_field('loc')[:,:]

x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]
x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]

s = x_train.shape[1]
h = s

x_train = x_train.reshape(ntrain,s, 1)
x_test = x_test.reshape(ntest,s, 1)
p_data = p_data.reshape(1, s, 1)
p_data = p_data / p_data[0,-1,0] * np.pi * 2

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

training_history = open('./grid_search_history/'+data_dist+'.txt', 'w')
training_history.write('Learning_Rate  Modes  Width  Weight_Decay --- Train_Loss Test_Loss \n')

for learning_rate in learning_rate_:
    for modes in modes_:
        for width in width_:
            for weight_decay in weight_decay_:
                # model
                model = FNO1d(modes, width).cuda()
                print(count_params(model))

                ################################################################
                # training and evaluation
                ################################################################
                

                optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                train_loss = np.zeros(epochs)
                myloss = LpLoss(size_average=False)
                for ep in range(epochs):
                    model.train()
                    t1 = default_timer()
                    train_mse = 0
                    train_l2 = 0
                    for x, y in train_loader:
                        x, y = x.cuda(), y.cuda()

                        optimizer.zero_grad()
                        out = model(x)

                        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
                        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
                        l2.backward() # use the l2 relative loss

                        optimizer.step()
                        train_mse += mse.item()
                        train_l2 += l2.item()

                    scheduler.step()
                    model.eval()
                    test_l2 = 0.0
                    with torch.no_grad():
                        for x, y in test_loader:
                            x, y = x.cuda(), y.cuda()

                            out = model(x)
                            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

                    train_mse /= len(train_loader)
                    train_l2 /= ntrain
                    test_l2 /= ntest

                    train_loss[ep] = train_l2

                    t2 = default_timer()
                print(str(learning_rate)+' '+ str(modes)+' '+ str(width)+' '+ str(weight_decay)+' --- '+ str(train_l2) +' '+ str(test_l2))
                training_history.write(str(learning_rate)+' '+ str(modes)+' '+ str(width)+' '+ str(weight_decay)+' --- '+ str(train_l2) +' '+ str(test_l2)+'\n')
training_history.close()