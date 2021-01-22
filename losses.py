import torch
from torch import nn

class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss


class NerfWLoss(nn.Module):
    def __init__(self, coef=1, lambda_u=0.01):
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u

    def forward(self, inputs, targets):
        coarse_color_loss = ((inputs['rgb_coarse']-targets)**2).mean()
        if 'rgb_fine' in inputs:
            fine_color_loss = ((inputs['rgb_fine']-targets)**2).mean()
            # fine_color_loss = \
            #     ((inputs['rgb_fine']-targets)**2 / (2*inputs['beta'].unsqueeze(1)**2)).mean()
            # beta_loss = torch.log(inputs['beta']).mean()
            # sigma_loss = self.lambda_u * inputs['transient_sigmas'].mean()

        return {'coarse_color_loss': self.coef * coarse_color_loss,
                'fine_color_loss': self.coef * fine_color_loss}

loss_dict = {'color': ColorLoss,
             'nerfw': NerfWLoss}