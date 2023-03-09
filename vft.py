"""
@author: Levi Lingsch

This code contains the classes and structures to initialize the Vandermonde matrices and perform the transformations in 1 or 2 dimensions on a torus.
"""

import torch
import numpy as np
import pdb

# class for 1-dimensional Fourier transforms on nonequispaced data
class vft1d:
    def __init__(self, positions, modes):
        self.modes = modes
        self.l = positions.shape[0]

        self.Vt, self.Vc = self.make_matrix()

    def make_matrix(self):
        V = torch.zeros([self.modes, self.l], dtype=torch.cfloat).cuda()
        for row in range(self.modes):
             for col in range(self.l):
                V[row, col] = np.exp(-1j * row *  self.positions[0,col,0]) 
        V = torch.divide(V, np.sqrt(self.l))

        return torch.transpose(V, 0, 1), torch.conj(V)

    def forward(self, data):
        return torch.matmul(data, self.Vt)

    def inverse(self, data):
        return torch.matmul(data, self.Vc)

# class for 2-dimensional Fourier transforms on a nonequispaced lattice of data
class vft2d:
    def __init__(self, x_positions, y_positions, x_modes, y_modes):
        self.x_positions = x_positions / (torch.max(x_positions)+1) * 2 * np.pi
        self.y_positions = y_positions / (torch.max(y_positions)+1) * 2 * np.pi
        self.x_modes = x_modes
        self.y_modes = y_modes
        self.x_l = x_positions.shape[0]
        self.y_l = y_positions.shape[0]

        self.Vxt, self.Vxc, self.Vyt, self.Vyc = self.make_matrix()

    def make_matrix(self):
        V_x = torch.zeros([self.x_modes, self.x_l], dtype=torch.cfloat).cuda()
        for row in range(self.x_modes):
             for col in range(self.x_l):
                V_x[row, col] = np.exp(-1j * row *  self.x_positions[col]) 
        V_x = torch.divide(V_x, np.sqrt(self.x_l))


        V_y = torch.zeros([2 * self.y_modes, self.y_l], dtype=torch.cfloat).cuda()
        for row in range(self.y_modes):
             for col in range(self.y_l):
                V_y[row, col] = np.exp(-1j * row *  self.y_positions[col]) 
                V_y[-(row+1), col] = np.exp(-1j * (self.y_l - row - 1) *  self.y_positions[col]) 
        V_y = torch.divide(V_y, np.sqrt(self.y_l))

        return torch.transpose(V_x, 0, 1), torch.conj(V_x), torch.transpose(V_y, 0, 1), torch.conj(V_y)

    def forward(self, data):
        data_fwd = torch.transpose(
                torch.matmul(
                    torch.transpose(
                        torch.matmul(data, self.Vxt)
                    , 2, 3)
                , self.Vyt)
                , 2,3)

        return data_fwd
    
    def inverse(self, data):
        data_inv = torch.transpose(
                torch.matmul(
                    torch.transpose(
                        torch.matmul(data, self.Vxc),
                    2, 3),
                self.Vyc),
                2, 3)
        
        return data_inv

# class for Double Fourier Sphere transforms on a nonequispaced lattice of data
class vdfs:
    def __init__(self, x_positions, y_positions, x_modes, y_modes):
        self.x_positions = x_positions #/ torch.max(x_positions) * 2 * np.pi
        self.y_positions = y_positions #/ torch.max(y_positions) * 2 * np.pi
        self.x_modes = x_modes
        self.y_modes = y_modes
        self.x_m = x_modes.shape[0]
        self.y_m = y_modes.shape[0]
        self.x_l = x_positions.shape[0]
        self.y_l = y_positions.shape[0]
        
        self.Vxt, self.Vxc, self.Vyt, self.Vyc = self.make_matrix()

    def make_matrix(self):
        V_x = torch.zeros([self.x_m, self.x_l], dtype=torch.cfloat).cuda()
        # V_x = torch.zeros([self.x_m, self.x_l], dtype=torch.float).cuda()
        for row in range(self.x_m):
             for col in range(self.x_l):
                V_x[row, col] = np.exp(-1j * self.x_modes[row] *  self.x_positions[col]) 
        V_x = torch.divide(V_x, np.sqrt(self.x_l))


        V_y = torch.zeros([self.y_m, self.y_l], dtype=torch.cfloat).cuda()
        # V_y = torch.zeros([self.y_m, self.y_l], dtype=torch.float).cuda()
        for row in range(self.y_m):
             for col in range(self.y_l):
                V_y[row, col] = np.exp(-1j * self.y_modes[row] *  self.y_positions[col])
        V_y = torch.divide(V_y, np.sqrt(self.y_l))

        return torch.transpose(V_x, 0, 1), torch.conj(V_x), torch.transpose(V_y, 0, 1), torch.conj(V_y)

    def forward(self, data):
        data_fwd = torch.transpose(
                torch.matmul(
                    torch.transpose(
                        torch.matmul(data, self.Vxt)
                    , 2, 3)
                , self.Vyt)
                , 2,3)

        return data_fwd
    
    def inverse(self, data):
        data_inv = torch.transpose(
                torch.matmul(
                    torch.transpose(
                        torch.matmul(data, self.Vxc),
                    2, 3),
                self.Vyc),
                2, 3)
        
        return data_inv

class fully_nonequispaced_vft:
    def __init__(self, x_positions, y_positions, modes):
        self.x_positions = x_positions
        self.y_positions = y_positions
        self.number_points = x_positions.shape[0]
        self.modes = modes

        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        forward_mat = torch.zeros((self.modes**2, self.number_points), dtype=torch.cfloat)
        for Y in range(self.modes):
            for X in range(self.modes):
                forward_mat[Y+X*self.modes, :] = np.exp(-1j* (X*self.x_positions[0]+Y*self.y_positions[0]))

        inverse_mat = torch.zeros((self.number_points, self.modes**2),  dtype=torch.cfloat)
        for Y in range(self.modes):
            for X in range(self.modes):
                inverse_mat[:, Y+X*self.modes] = np.exp(1j* (X*self.x_positions[0]+Y*self.x_positions[0]))
        # reconstruction_flat = (np.matmul(inverse_mat, Fourier_3dmatmul) / np.sqrt(number_points)).real
        return forward_mat, inverse_mat

    def forward(self, data):
        data_fwd = (np.matmul(self.V_fwd, data) / np.sqrt(self.number_points))
        data_fwd = torch.reshape(data_fwd, (data_fwd.shape[0], data_fwd.shape[1], self.modes, self.modes))

        return data_fwd

    def inverse(self, data):
        data = torch.reshape(data, (data.shape[0], data.shape[1], self.modes**2, 1))
        data_inv = (np.matmul(self.V_inv, data) / np.sqrt(self.number_points)).real
        
        return data_inv