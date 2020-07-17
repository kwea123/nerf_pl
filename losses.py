import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss
               

class NormalLoss(nn.Module):
    def __init__(self, lamb=1e-2):
        super().__init__()
        self.lamb = lamb

    def forward(self, inputs):
        loss = (1-(inputs['normals_ndc_fine']*inputs['normals_ndc_neighbors_fine']).sum(1)).mean()

        return self.lamb * loss

loss_dict = {'mse': MSELoss,
             'normal': NormalLoss}