"""
@author: Levi Lingsch

This code contains the classes and structures to initialize the Vandermonde matrices and perform the transformations in 1 or 2 dimensions on a torus.
"""

import torch
import numpy as np

class vft1d:
    def __init__(self, positions, modes):
        self.positions = positions
        self.modes = modes
        self.l = positions.shape[0]

        self.Vt, self.Vc = self.make_matrix()

    def make_matrix(self):
        V = torch.zeros([self.modes1, self.l], dtype=torch.cfloat).cuda()
        for row in range(self.modes1):
             for col in range(self.l):
                V[row, col] = np.exp(-1j * row *  self.positions[col]) 
        V = torch.divide(V, np.sqrt(self.l))

        return torch.transpose(V, 0, 1), torch.conj(V)

    def forward(self, data):
        return torch.matmul(data, self.Vt)

    def inverse(self, data):
        return torch.matmul(data, self.Vc)