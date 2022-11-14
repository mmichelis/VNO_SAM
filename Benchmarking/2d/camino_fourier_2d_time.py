"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.interpolate
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
        self.fc0 = nn.Linear(12, self.width)
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
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################
data_dist = 'cc'
interp = 'cubic'
file_path = '../../../VNO_data/2d/'

ntrain = 64
ntest = 64

modes = 12
width = 20

batch_size = 1
batch_size2 = batch_size

epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

growth = 1.75
offset = 20 # rip takeoff

path = f'{data_dist}_ns_gr{growth}_off{offset}_ep{epochs}_m{modes}_w{width}'

sub = 1
T_in = 10
T = 10
step = 1

################################################################
# load data
################################################################

t1 = default_timer()
print('Preprocessing Data...')

def load_data():
    TRAIN_PATH = f'{file_path}navierstokes_512_512_v1e-4_{0}.mat'
    reader = MatReader(TRAIN_PATH)
    test_a = reader.read_field('vorticity')[:,:,:,:T_in]
    test_u = reader.read_field('vorticity')[:,:,:,T_in:T+T_in]

    TRAIN_PATH = f'{file_path}navierstokes_512_512_v1e-4_{1}.mat'
    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field('vorticity')[:,:,:,:T_in]
    train_u = reader.read_field('vorticity')[:,:,:,T_in:T+T_in]
    # for NUM in range(2, 2):
    #     TRAIN_PATH = f'{file_path}navierstokes_512_512_v1e-4_{NUM}.mat'
    #     reader = MatReader(TRAIN_PATH)
    #     train_a = torch.cat((train_a, reader.read_field('vorticity')[:,:,:,:T_in]))
    #     train_u = torch.cat((train_u, reader.read_field('vorticity')[:,:,:,T_in:T+T_in]))

    return test_a, test_u, train_a, train_u
test_a, test_u, train_a, train_u = load_data()

# define the lattice of points to select for the simulation
def define_positions(growth, offset):
    # the bottom and left boundaries are both at 0, but not the top or right boundaries
    top = 512
    right = 512

    # the data should already be centered longitudinally
    center_x = right//2
    center_y = top//2

    # define the bounds of the equispaced region
    side_s = center_y - offset
    side_n = center_y + offset
    side_w = center_x - offset
    side_e = center_x + offset

    # calculate the number of points in each side of the nonequispaced region
    num_s = np.floor(side_s**(1/growth))
    num_n = num_s
    num_w = np.floor(side_w**(1/growth))
    num_e = num_w #np.floor((right - side_e)**(1/growth))

    # define the positions of points to each side
    points_s = torch.flip(side_s - torch.round(torch.arange(num_s+1)**growth), [0])
    points_n = side_n + torch.round(torch.arange(num_n)**growth)
    points_w = torch.flip(side_w - torch.round(torch.arange(num_w+1)**growth),[0])
    points_e = side_e + torch.round(torch.arange(num_e)**growth)

    # positions with equispaced distributions
    central_lat = torch.arange(side_s+1, side_n)
    central_lon = torch.arange(side_w+1, side_e)

    # fix positions together
    lat = torch.cat((points_s, central_lat, points_n))
    lon = torch.cat((points_w, central_lon, points_e))
    return lon.int(), lat.int()
x_pos, y_pos = define_positions(growth, offset)

def make_sparse(test_a, test_u, train_a, train_u, x_pos, y_pos):
    test_a = torch.index_select(torch.index_select(test_a, 1, x_pos), 2, y_pos)
    test_u = torch.index_select(torch.index_select(test_u, 1, x_pos), 2, y_pos)
    train_a = torch.index_select(torch.index_select(train_a, 1, x_pos), 2, y_pos)
    train_u = torch.index_select(torch.index_select(train_u, 1, x_pos), 2, y_pos)

    return test_a, test_u, train_a, train_u
test_a, test_u, train_a, train_u = make_sparse(test_a, test_u, train_a, train_u, x_pos, y_pos)

def interpolate_positions(data, x_pos, y_pos, method='linear'):
    x_pos = x_pos.numpy()
    y_pos = y_pos.numpy()
    dx, dy = np.meshgrid(x_pos, y_pos)
    sparse_loc = np.stack((dx.flatten(), dy.flatten()), axis=1)

    x = np.arange(np.min(x_pos), np.max(x_pos)+1,1)
    y = np.arange(np.min(y_pos), np.max(y_pos)+1,1)
    dx, dy = np.meshgrid(x, y)
    dense_loc = np.stack((dx.flatten(), dy.flatten()), axis=1)
    full_dense_data = np.zeros([data.shape[0], x.shape[0], y.shape[0], data.shape[-1]], dtype=np.float32)
    for id in range(data.shape[0]):
        for time in range(data.shape[-1]):
            # flatten the data
            sparse_data = data[id, :, :, time].numpy().flatten()
            dense_data = scipy.interpolate.griddata(sparse_loc, sparse_data, dense_loc, method=method)
            dense_data = dense_data.reshape(x.shape[0],y.shape[0])
            full_dense_data[id, :, :, time] = dense_data
    return torch.from_numpy(full_dense_data)
start_interp = default_timer()
train_a = interpolate_positions(train_a, x_pos, y_pos)
train_a = interpolate_positions(train_u, x_pos, y_pos)
test_a = interpolate_positions(test_a, x_pos, y_pos)
test_u = interpolate_positions(test_u, x_pos, y_pos)
stop_interp = default_timer()
print(f'interpolation time of {stop_interp-start_interp} for 2 vorticity sample files.') # about 80 seconds

# pdb.set_trace()
# S_x = torch.max(x_pos)
# S_y = torch.max(y_pos)

# assert (S_x == train_u.shape[3])
# assert (S_y == train_u.shape[-2])
# assert (T == train_u.shape[-1])


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print(f'Processing finished in {t2-t1} seconds.')
################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width).cuda()
# model = torch.load(path_model)


print(count_params(model))
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)

training_history = open('./training_history/'+interp+'_from_'+data_dist+'.txt', 'w')
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

        # pdb.set_trace()
        # yy = torch.index_select(yy, 1, x_pos)
        # yy = torch.index_select(yy, 2, y_pos)

        this_batch_size = xx.shape[0]
        for t in range(0, T, step):
            y = yy[..., t:t + step]

            im = model(xx)
            # im = torch.index_select(im, 1, x_pos)
            # im = torch.index_select(im, 2, y_pos)
            pdb.set_trace()
            loss += myloss(im.reshape(this_batch_size, -1), y.reshape(this_batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            # xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(this_batch_size, -1), yy.reshape(this_batch_size, -1))
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

            # pdb.set_trace()
            yy = torch.index_select(yy, 1, x_pos)
            yy = torch.index_select(yy, 2, y_pos)

            this_batch_size = xx.shape[0]
            for t in range(0, T, step):
                y = yy[..., t:t + step]

                im = model(xx)
                im = torch.index_select(im, 1, x_pos)
                im = torch.index_select(im, 2, y_pos)
                
                loss += myloss(im.reshape(this_batch_size, -1), y.reshape(this_batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                # xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(this_batch_size, -1), yy.reshape(this_batch_size, -1)).item()

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
prediction_history = open('./training_history/'+interp+'_from_'+data_dist+'_test_loss.txt', 'w')
batch_size=1        # need to set this otherwise the loss outputs are not correct
with torch.no_grad():
    for xx, yy in test_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        
        yy = torch.index_select(yy, 1, x_pos)
        yy = torch.index_select(yy, 2, y_pos)
        
        full_pred = model(xx)
        full_pred = torch.index_select(full_pred, 1, x_pos)
        full_pred = torch.index_select(full_pred, 2, y_pos)
        for t in range(0, T, step):
            y = yy[..., t:t + step]

            im = model(xx)
            im = torch.index_select(im, 1, x_pos)
            im = torch.index_select(im, 2, y_pos)

            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            # xx = torch.cat((xx[..., step:], im), dim=-1)

        print(index, loss.item() / T)
        index = index + 1
        full_pred = torch.cat((full_pred, pred), -1)

        prediction_history.write(str(loss.item() / T)+'\n')
    prediction_history.close()

# ll: save as .txt instead of .mat
scipy.io.savemat('./predictions/'+path+'.mat', mdict={'pred': full_pred.cpu().numpy()})