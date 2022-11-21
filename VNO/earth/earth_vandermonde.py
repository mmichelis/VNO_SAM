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
from vft import *
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
        # self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        # self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.float))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.float))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # batchsize = x.shape[0]
        # pdb.set_trace()
        # x_ft = transformer.forward(x.cfloat())
        x_ft = transformer.forward(x)

        # Multiply relevant Fourier modes
        # out_ft = torch.zeros(batchsize, self.out_channels,  2 * self.modes1, self.modes2, dtype=torch.cfloat, device=x.device)
        x_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        x_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        # x_ft[:, :, :self.modes1, :self.modes2] = torch.matmul(x_ft[:, :, :self.modes1, :self.modes2], self.weights1[:,0,:,:])
        # x_ft[:, :, -self.modes1:, :self.modes2] = torch.matmul(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2[:,0,:,:])

        #Return to physical space
        # x = transformer.inverse(x_ft).real
        x = transformer.inverse(x_ft)

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
        self.fc0 = nn.Linear(T_in+2, self.width)
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
        gridx = lon
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_y, 1, 1])
        # gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = lat
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################

selected_modes = np.concatenate((np.arange(20), np.arange(20,50,3)))
# selected_modes = np.arange(16)
print(f'selected modes: {selected_modes}')
modes = selected_modes.shape[0]
width = 40

batch_size = 10
batch_size2 = batch_size

epochs = 200
learning_rate = 0.0025
scheduler_step = 10
scheduler_gamma = 0.90

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

DAT = 'QLML'
path = DAT+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)

runtime = np.zeros(2, )
t1 = default_timer()

T_in = 18
T = 6
step = 1

center_lon = int(188 * 1.6)
center_lat = 137 * 2
growth = 1.4
offset = 30

left = center_lon - offset
right = center_lon + offset
bottom = center_lat - offset
top = center_lat + offset
##############################################################
# load data
################################################################
# Due to the amount of data required for this project, it is necessary to construct the sparse data directly within this code. There is not enough storage elsewhere.
def load_data():
    TEST_PATH = f'../../../VNO_data/EarthData/{DAT}_data_0.mat'
    reader = MatReader(TEST_PATH)
    test_a = reader.read_field(DAT)[:,:T_in,:,:]
    test_u = reader.read_field(DAT)[:,T_in:T+T_in,:,:]

    TRAIN_PATH = f'../../../VNO_data/EarthData/{DAT}_data_1.mat'
    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field(DAT)[:,:T_in,:,:]
    train_u = reader.read_field(DAT)[:,T_in:T+T_in,:,:]

    for NUM in range(2, 5):
        TRAIN_PATH = f'../../../VNO_data/EarthData/{DAT}_data_{NUM}.mat'
        reader = MatReader(TRAIN_PATH)
        train_a = torch.cat((train_a, reader.read_field(DAT)[:,:T_in,:,:]))
        train_u = torch.cat((train_u, reader.read_field(DAT)[:,T_in:T+T_in,:,:]))

    return test_a, test_u, train_a, train_u
test_a, test_u, train_a, train_u = load_data()
# shape at this point: [ntrain/ntest, 12, 361, 576]

# I am concatenating several large data file together here, so the ntrain is variable. Should just reset it here with the actual value.
ntrain = train_a.shape[0]
ntest = test_a.shape[0]
print(train_u.shape)
print(test_u.shape)

# the data must be centered longitudinally so that it stays on a lattice when doubled
def center_longitutude(data, center):
    lon_pts = data.shape[-1]
    return torch.cat((data[:,:,:,center-lon_pts//2:], data[:,:,:,:center-lon_pts//2]), -1)
test_a = center_longitutude(test_a, center_lon)
test_u = center_longitutude(test_u, center_lon)
train_a = center_longitutude(train_a, center_lon)
train_u = center_longitutude(train_u, center_lon)

# define the lattice of points to select for the simulation
def define_positions(center_lat, growth, offset):
    # the bottom and left boundaries are both at 0, but not the top or right boundaries
    top = 180 * 2
    right = 360 * 1.6

    # the data should already be centered longitudinally
    center_lon = right//2

    # define the bounds of the equispaced region
    side_s = center_lat - offset
    side_n = center_lat + offset
    side_w = center_lon - offset
    side_e = center_lon + offset

    # calculate the number of points in each side of the nonequispaced region
    num_s = np.floor(side_s**(1/growth))+1
    num_n = np.floor((top - side_n)**(1/growth))+1
    num_w = np.floor(side_w**(1/growth))
    num_e = num_w

    # define the positions of points to each side
    points_s = torch.flip(side_s - torch.round(torch.arange(num_s)**growth), [0])
    points_n = side_n + torch.round(torch.arange(num_n)**growth)
    points_w = torch.flip(side_w - torch.round(torch.arange(num_w)**growth),[0])
    points_e = side_e + torch.round(torch.arange(num_e)**growth)

    # print(f"east {num_e} west {num_w}")

    # positions with equispaced distributions
    central_lat = torch.arange(side_s+1, side_n)
    central_lon = torch.arange(side_w+1, side_e)

    # fix positions together
    lat = torch.cat((points_s, central_lat, points_n))
    lon = torch.cat((points_w, central_lon, points_e))
    return lon.int(), lat.int(), num_w, num_s
lon, lat, num_w, num_s = define_positions(center_lat, growth, offset)


# select the positions from the desired distribution and double accordingly
def double_data(data, lon, lat):
    sparse_data = torch.index_select(torch.index_select(data, -2, lat), -1, lon)
    double_data = sparse_data
    double_data = torch.cat((torch.flip(sparse_data, [-2,-1]), sparse_data), -2)
    return double_data
test_a = double_data(test_a, lon, lat)
test_u = double_data(test_u, lon, lat)
train_a = double_data(train_a, lon, lat)
train_u = double_data(train_u, lon, lat)
# shape at this point: [ntrain/ntest, 12, 194, 123]
print(train_u.shape)
print(test_u.shape)

# scale and modify the lon / lat as needed
lon = lon * np.pi / 180 / 1.6
lat = np.pi - lat * np.pi / 180 / 2
lat = torch.cat((torch.flipud(lat), 2*np.pi - lat), 0)


# can't assert without knowing shapes beforehand, so I just gather them from the data and use them where necessary
S_x = train_u.shape[-1]
S_y = train_u.shape[-2]
assert (T == train_u.shape[1])

# NOTE: using reshape will severely alter the data. Use torch. swapaxes instead to get it into the correct format.
# go from (0, 1, 2, 3) to (0, 3, 2, 1)
# train_a = train_a.reshape(ntrain,S_x,S_y,T_in)
# test_a = test_a.reshape(ntest,S_x,S_y,T_in)
# reshape the data to be in [Number of exmaples, X-coordinates, Y-coordinates, 1, Time], this is how their code was originally written
train_a = torch.swapaxes(torch.swapaxes(train_a, 1, 3), 1, 2)
train_u = torch.swapaxes(torch.swapaxes(train_u, 1, 3), 1, 2)
test_a = torch.swapaxes(torch.swapaxes(test_a, 1, 3), 1, 2)
test_u = torch.swapaxes(torch.swapaxes(test_u, 1, 3), 1, 2)
# shape at this point: [ntrain/ntest, 123, 194, 12]


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')


# normalize the data via min-max normalization


################################################################
# training and evaluation
################################################################

model = FNO2d(modes, modes, width).cuda()
transformer = vdfs(lon, lat, selected_modes, selected_modes)

print(count_params(model))
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)

training_history = open('./training_history/2d_vandermonde.txt', 'w')
training_history.write('Epoch  Time  Train_L2_Step  Train_L2_Full  Test_L2_Step  Test_L2_Full  \n')

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
                y = yy[:, num_w:num_w+2*offset, num_s:num_s+2*offset, t:t + step]
                pdb.set_trace()
                full_im = model(xx)
                im = im[:, num_w:num_w+2*offset, num_s:num_s+2*offset]
                asd,jkl = np.mgrid[0:2*offset, 0:2*offset]
                plt.contourf(asd, jkl, y)
                plt.show()
                plt.contourf(asd, jkl, im)
                plt.show()
                
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], full_im), dim=-1)

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




# pred = torch.zeros(test_u.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
prediction_history = open('./training_history/2d_vandermonde_test_loss.txt', 'w')
batch_size = 1
with torch.no_grad():
    for xx, yy in test_loader:
        step_loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        
        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)

            step_loss += myloss(im.reshape(1, -1), y.reshape(1, -1))
            
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        full_loss = myloss(pred.reshape(1, -1), yy.reshape(1, -1))

        print(index, full_loss.item(), step_loss.item() / T)
        index = index + 1
        prediction_history.write(f'{full_loss.item()}   {step_loss.item() / T} \n')
prediction_history.close()
print(pred.shape)

# only save one prediction to keep space low
scipy.io.savemat('./predictions/'+path+'.mat', mdict={'pred': pred.cpu().numpy(), 'lat': lat.cpu().numpy(), 'lon': lon.cpu().numpy()})