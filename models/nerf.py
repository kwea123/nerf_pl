import torch
from torch import nn

class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self, typ,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, in_channels_t=16,
                 skips=[4],
                 beta_min=0.03):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_t: number of input channels for t
        skips: add skip connection in the Dth layer
        beta_min: minimum pixel color variance (used only for fine model)
        """
        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_t = in_channels_t
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Linear(W, 1)
        self.static_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())

        # if typ == 'fine':
        #     self.beta_min = beta_min
        #     # transient encoding layers (maybe more layers?)
        #     self.transient_encoding = nn.Sequential(
        #                                 nn.Linear(W+in_channels_t, W//2),
        #                                 nn.ReLU(True),
        #                                 nn.Linear(W//2, W//2), nn.ReLU(True))
        #     # transient output layers
        #     self.transient_sigma = nn.Sequential(nn.Linear(W//2, 1), nn.ReLU(True))
        #     self.transient_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
        #     self.transient_beta = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())

    def forward(self, x, sigma_only=False, has_transient=True):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir+self.in_channels_t))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if sigma_only:
            input_xyz = x
        elif has_transient:
            input_xyz, input_dir, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir,
                                self.in_channels_t], dim=-1)
        else:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
            

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        static_sigma = self.static_sigma(xyz_) # (B, 1)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)

        return static

        # if not has_transient:
        #     return static

        # transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        # transient_encoding = self.transient_encoding(transient_encoding_input)
        # transient_sigma = self.transient_sigma(transient_encoding) # (B, 1)
        # transient_rgb = self.transient_rgb(transient_encoding) # (B, 3)
        # transient_beta = self.beta_min + self.transient_beta(transient_encoding) # (B, 1)

        # transient = torch.cat([transient_rgb,
        #                        transient_sigma,
        #                        transient_beta], 1) # (B, 5)

        # return torch.cat([static, transient], 1) # (B, 9)