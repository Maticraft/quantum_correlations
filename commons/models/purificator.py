import numpy as np
import torch.nn as nn


class Purificator(nn.Module):
    def __init__(self, qbits_num, out_ch_per_rho, filters_ratio, kernel_size, hidden_convs, output_dim = 2, dilation = 1, ratio_type = None, padding = 0):
        super(Purificator, self).__init__()

        self.input_channels = 2
        self.dim = 2**qbits_num
        self.kernel_size = kernel_size
        self.qbits_num = qbits_num
        self.ocpr = out_ch_per_rho
        self.output_dim = output_dim

        in_ch = self.input_channels
        fr = filters_ratio
        self.out_dim = self.dim

        convs = []
        i = 0
        while (self.out_dim + 2*padding - dilation*(self.kernel_size - 1)) >= self.output_dim and i < hidden_convs:
            out_ch = int(in_ch*fr)
            convs.append(nn.Conv2d(in_ch, out_ch, self.kernel_size, dilation = dilation, padding= padding).double())
            convs.append(nn.ReLU())
            in_ch = out_ch

            if ratio_type == 'sqrt':
                fr = np.sqrt(fr)
            elif ratio_type == 'sq':
                fr = fr ** 2

            self.out_dim += 2*padding - dilation*(self.kernel_size - 1)
            i += 1

        self.out_ch = out_ch
        self.convs = nn.Sequential(*convs)

        ks = int((self.out_dim + 2*padding - self.output_dim) / dilation) + 1
        self.out_dim += 2*padding - dilation*(ks - 1)
        assert self.out_dim == self.output_dim, "Wrong output dimension"

        self.output_conv = nn.Conv2d(in_ch, 2*self.ocpr, ks, dilation = dilation, padding = padding)


    def forward(self, x):
        x = self.convs(x)
        output = self.output_conv(x)

        return output