from math import floor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# SeparatorBipart model -- NN, which tries to learn density matrices of separate subsystems (with division made for 1|N-1 subsystems only) 
# - if it is possible -> state is separable, if not, then it is entangled 
class FancySeparatorBipart(nn.Module):
    def __init__(self, qbits_num, out_ch_per_rho):
        super(FancySeparatorBipart, self).__init__()

        self.input_channels = 2
        self.dim = 2**qbits_num
        self.qbits_num = qbits_num
        self.ocpr = out_ch_per_rho
        self.output_channels = self.input_channels*self.ocpr
        self.biparts_num = self.qbits_num

        self.conv, self.conv_prim = self.make_convs(self.qbits_num, self.ocpr, self.input_channels, self.output_channels)

    # simple for 1|N-1 biparts only
    def make_convs(self, qubits_num, ocpr, in_channels, out_channels):
        assert qubits_num >= 2, "Wrong number of qubits"
        ch_multiplier = ocpr
        in_ch = in_channels
        out_ch = int(in_ch*ch_multiplier)
        dim = 2**qubits_num
        small_dim = 2 # constant for 1|N-1
        large_dim = dim // small_dim

        conv = nn.Conv2d(in_ch, out_channels, kernel_size= large_dim, stride= large_dim) #conv applied to obtain first matrices
        conv_prim = nn.Conv2d(in_ch, out_channels, kernel_size= small_dim, stride= 1, dilation = large_dim) #conv applied to obtain second (complementary) matrices

        return conv, conv_prim


    def forward(self, x):
        """
        output dims:
            0 - separate matrices (e.g. of dim 2 and 4); 
            2 - examples;    
            3 - channels;    
            4, 5 - density matrix   
        """
        output = [torch.sigmoid(self.conv(x)), torch.sigmoid(self.conv_prim(x))]

        return output


# SeparatorBipart model -- NN, which tries to learn density matrices of separate subsystems (with division made for two subsystems only) 
# - if it is possible -> state is separable, if not, then it is entangled
class SeparatorBipart(nn.Module):
    def __init__(self, qbits_num, out_ch_per_rho, filters_ratio, kernel_size, output_dims = (2, 4), dilation = 1, ratio_type = None, padding = 0):
        super(SeparatorBipart, self).__init__()

        self.input_channels = 2
        self.dim = 2**qbits_num
        self.kernel_size = kernel_size
        self.qbits_num = qbits_num
        self.ocpr = out_ch_per_rho
        self.output_dims = output_dims

        self.biparts_num = self.qbits_num

        in_ch = self.input_channels
        fr = filters_ratio
        self.out_dim = self.dim

        convs = []
        while (self.out_dim - dilation*(self.kernel_size - 1)) > max(self.output_dims):
            out_ch = int(in_ch*fr)
            convs.append(nn.Conv2d(in_ch, out_ch, self.kernel_size, dilation = dilation, padding= padding).double())
            convs.append(nn.ReLU())
            in_ch = out_ch

            if ratio_type == 'sqrt':
                fr = np.sqrt(fr)
            elif ratio_type == 'sq':
                fr = fr ** 2

            self.out_dim += 2*padding - dilation*(self.kernel_size - 1)

        self.out_ch = out_ch
        self.convs = nn.Sequential(*convs)

        ks0 = int((self.out_dim - self.output_dims[0]) / dilation) + 1
        self.out_dim0 = self.out_dim + 2*padding - dilation*(ks0 - 1)
        assert self.out_dim0 == self.output_dims[0], "Wrong output dimension 0"

        ks1 = int((self.out_dim - self.output_dims[1]) / dilation) + 1
        self.out_dim1 = self.out_dim + 2*padding - dilation*(ks1 - 1)
        assert self.out_dim1 == self.output_dims[1], "Wrong output dimension 1"

        self.output_convs = nn.ModuleList([nn.Conv2d(in_ch, 2*self.ocpr, ks0, dilation = dilation), nn.Conv2d(in_ch, 2*self.ocpr, ks1, dilation = dilation)])


    def forward(self, x):
        """
        output dims:
            0 - biparts
            1 - separate matrices (e.g. of dim 2 and 4)
            2 - examples
            3 - channels
            4, 5 - density matrix
        """
        x = self.convs(x)
        output = [out_conv(x) for out_conv in self.output_convs]

        return output


# Fancy Separator model -- NN, which tries to learn density matrices of separate subsystems - if it is possible -> state is separable,
# if not, then it is entangled
# This model applies different convolutions in order to receive different density matrices for separate qubits 
class FancySeparator(nn.Module):
    def __init__(self, qbits_num, out_ch_per_rho, input_channels = 2, fc_layers = 0):
        super(FancySeparator, self).__init__()

        self.input_channels = input_channels
        self.dim = 2**qbits_num
        self.qbits_num = qbits_num
        self.ocpr = out_ch_per_rho
        self.output_channels = self.input_channels*self.ocpr
        self.fc_layers_num = fc_layers

        self.convs, self.convs_prim = self.make_convs(self.qbits_num, self.ocpr, self.input_channels, self.output_channels)
        if self.fc_layers_num > 0:
            self.fc_layers = nn.ModuleList(
                [
                    nn.ModuleList(
                        [nn.Linear(self.output_channels*2*2, self.output_channels*2*2) for i in range(fc_layers)]
                    ) for j in range(self.qbits_num)
                ]
            )

    def make_convs(self, qubits_num, ocpr, in_channels, out_channels):
        assert qubits_num >= 2, "Wrong number of qubits"
        ch_multiplier = ocpr**(1/(qubits_num-1))
        in_ch = in_channels
        out_ch = int(in_ch*ch_multiplier)

        convs = []
        convs_prim = []
        for i in range(1, qubits_num - 1):
            curr_dim = 2**(qubits_num - i)
            convs.append(nn.Conv2d(in_ch, out_channels, kernel_size= curr_dim, stride= curr_dim))
            convs_prim.append(nn.Conv2d(in_ch, out_ch, kernel_size= 2, dilation=curr_dim))

            in_ch = out_ch
            out_ch = int(in_ch*ch_multiplier)

        convs.append(nn.Conv2d(in_ch, out_channels, kernel_size= 2, stride= 2))
        convs.append(nn.Conv2d(in_ch, out_channels, kernel_size= 2, dilation=2))

        return nn.ModuleList(convs), nn.ModuleList(convs_prim)


    def forward(self, x, noise = False):
        x_temp = x
        output = [self.convs[0](x_temp)]
        for i in range(1, self.qbits_num - 1):
            x_temp = self.convs_prim[i-1](x_temp)
            output.append(self.convs[i](x_temp))
        output.append(self.convs[self.qbits_num - 1](x_temp))

        if noise:
            probs = self.noise(x)
            for i in range(self.qbits_num):
                for j in range(self.ocpr):
                    output[i][:, j, :, :] *= probs[:, i, j].view(-1, 1, 1)
                    output[i][:, self.ocpr + j, :, :] *= probs[:, i, j].view(-1, 1, 1)

        if self.fc_layers_num > 0:
            for i in range(self.qbits_num):
                output[i] = output[i].view(-1, self.output_channels*2*2)
                for j in range(self.fc_layers_num):
                    output[i] = F.relu((output[i]))
                    output[i] = self.fc_layers[i][j](output[i])
                output[i] = output[i].view(-1, self.output_channels, 2, 2)

        return output


    def noise(self, x):
        noise = torch.randn(x.shape[0], self.qbits_num, self.ocpr).to(x.device)
        noise = F.softmax(noise, dim = 2)
        return noise


class SiameseFancySeparator(FancySeparator):
    def __init__(self, qbits_num, sep_ch, input_channels = 2):
        super(SiameseFancySeparator, self).__init__(qbits_num, sep_ch, input_channels)

    def forward(self, x1, x2):
        sep_mats1 = super().forward(x1)
        sep_mats2 = super().forward(x2)
        return sep_mats1, sep_mats2


# Separator model -- NN, which tries to learn density matrices of separate subsystems - if it is possible -> state is separable,
# if not, then it is entangled
class Separator(nn.Module):
    def __init__(self, qbits_num, out_ch_per_rho, filters_ratio, kernel_size, output_dim = 2, dilation = 1, ratio_type = None, padding = 0, stride = 1, input_channels = 2):
        super(Separator, self).__init__()

        self.input_channels = input_channels
        self.dim = 2**qbits_num
        self.kernel_size = kernel_size
        self.qbits_num = qbits_num
        self.ocpr = out_ch_per_rho
        self.output_dim = output_dim

        in_ch = self.input_channels
        fr = filters_ratio
        self.out_dim = self.dim

        convs = []
        while (floor((self.out_dim - dilation*(self.kernel_size - 1) - 1)/stride + 1) > self.output_dim):
            out_ch = int(in_ch*fr)
            convs.append(nn.Conv2d(in_ch, out_ch, self.kernel_size, stride, dilation = dilation, padding= padding).double())
            convs.append(nn.ReLU())
            in_ch = out_ch

            if ratio_type == 'sqrt':
                fr = np.sqrt(fr)
            elif ratio_type == 'sq':
                fr = fr ** 2

            self.out_dim = floor((self.out_dim + 2*padding - dilation*(self.kernel_size - 1) - 1)/stride + 1)

        self.out_ch = out_ch
        self.convs = nn.Sequential(*convs)

        ks = int(-(stride*(self.conv_out_dim - 1) + 1 - self.conv_out_dim)/dilation + 1)

        self.out_dim = floor((self.out_dim + 2*padding - dilation*(ks - 1) - 1)/stride + 1)
        assert self.out_dim == self.output_dim, "Wrong output dimension"

        self.output_convs = nn.ModuleList([nn.Conv2d(in_ch, self.input_channels*self.ocpr, ks, dilation = dilation) for i in range(self.qbits_num)])


    def forward(self, x):
        x = self.convs(x)
        output = [out_conv(x) for out_conv in self.output_convs]

        return output


def rho_reconstruction(x, separator_output):
    ch = separator_output[0].size()[1] // 2
    rho = torch.zeros_like(x[:,0,:,:], dtype = torch.cdouble)

    for i in range(ch):
        dms = separator_output[0]
        rho_real = dms[:, i, :, :]
        rho_imag = dms[:, ch + i, :, :]
        rho_i = torch.complex(rho_real, rho_imag)

        for j in range(1, len(separator_output)):
            dms = separator_output[j]
            ch = dms.size()[1] // 2
            rho_real = dms[:, i, :, :]
            rho_imag = dms[:, ch + i, :, :]
            rho_j = torch.complex(rho_real, rho_imag)
            rho_i = torch.stack([torch.kron(rho_i[k], rho_j[k]) for k in range(len(rho_i))])
        rho += rho_i

    rho = rho / ch
    rho = torch.stack((rho.real, rho.imag), dim = 1)
    return rho
