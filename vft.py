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
        self.x_positions = x_positions / torch.max(x_positions) * 2 * np.pi
        self.y_positions = y_positions / torch.max(y_positions) * 2 * np.pi
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
        self.x_positions = x_positions / torch.max(x_positions) * 2 * np.pi
        self.y_positions = y_positions / torch.max(y_positions) * 2 * np.pi
        self.x_modes = x_modes
        self.y_modes = y_modes
        self.x_m = x_modes.shape[0]
        self.y_m = y_modes.shape[0]
        self.x_l = x_positions.shape[0]
        self.y_l = y_positions.shape[0]
        
        self.Vxt, self.Vxc, self.Vyt, self.Vyc = self.make_matrix()

    def make_matrix(self):
        V_x = torch.zeros([self.x_m, self.x_l], dtype=torch.cfloat).cuda()
        for row in range(self.x_m):
             for col in range(self.x_l):
                V_x[row, col] = np.exp(-1j * self.x_modes[row] *  self.x_positions[col]) 
                # V_x[row, col] = np.cos((2*row+1) *  self.x_positions[col] / 2) 
        V_x = torch.divide(V_x, np.sqrt(self.x_l))


        V_y = torch.zeros([2*self.y_m, self.y_l], dtype=torch.cfloat).cuda()
        for row in range(self.y_m):
             for col in range(self.y_l):
                V_y[row, col] = np.exp(-1j * self.y_modes[row] *  self.y_positions[col]) 
                V_y[-(row+1), col] = np.exp(-1j * (self.y_l - self.y_modes[row]) *  self.y_positions[col]) 
                # V_y[row, col] = np.sin((2*self.y_modes[row]) *  self.y_positions[col] / 2) 
                # V_y[-row, col] = np.sin((2*(self.y_l - self.y_modes[row])) *  self.y_positions[col] / 2) 
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