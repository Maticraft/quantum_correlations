import numpy as np
from qiskit.quantum_info import DensityMatrix, partial_trace
import torch
import torch.nn as nn


from math import floor

import torch.nn.functional as F

from commons.metrics import bipartitions_num


# General model consisting of:
# - conv_num convolutional layers, each with kernel_size, dilation and number of output channels based on input channels and filters_ratio as well as ratio_type,
#       each such layer is followed by ReLU activation function and pooling layer depending on pooling parameter
# - fc_num fully connected layers, each consisting of 128 units and ReLU activation function,
#       to each such layer dropout can be added if it is set to True
# - finally the network can work both in regression and classifier mode with output_size
# - to make the model scalable for different number of qubits, qbits_num is also needed
class CNN(nn.Module):
    def __init__(self, qbits_num, output_size, conv_num, fc_num, kernel_size, filters_ratio, dilation = 1, ratio_type = None, pooling = None, dropout = False, mode = 'regression', upsample_layers = 0, fc_trans = "const", fc_units = 128, input_channels = 2, stride = 1, activation = 'relu', cn_in_block = 1, batch_norm = False):
        super(CNN, self).__init__()

        self.input_channels = input_channels
        self.dim = 2**qbits_num
        self.mode = mode
        self.conv_out_dim = self.dim
        self.kernel_size = kernel_size

        in_ch = self.input_channels

        fr = filters_ratio
        if upsample_layers > 0:
            upconvs = []
            up_out_chs = [in_ch]

            for i in range(0, upsample_layers):
                out_ch = int(in_ch*fr)
                upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride).double())
                if batch_norm:
                    upconvs.append(nn.BatchNorm2d(out_ch))
                if activation == 'relu':
                    upconvs.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    upconvs.append(nn.LeakyReLU(0.1))

                in_ch = out_ch

                if ratio_type == 'sqrt':
                    fr = np.sqrt(fr)
                elif ratio_type == 'sq':
                    fr = fr ** 2

                self.conv_out_dim = (self.conv_out_dim - 1)*stride + self.kernel_size
                up_out_chs.append(out_ch)

            self.upconvs = nn.ModuleList(upconvs)
            self.upconv_num = upsample_layers
        else:
            self.upconv_num = 0

        convs = []
        for i in range(0, conv_num):
            out_ch = int(in_ch*fr)

            if cn_in_block > 1:
                if upsample_layers > 0:
                    raise ValueError("Larger conv blocks are not compatible with upsample layers")
                cn_block = []
                cn_block.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride,  dilation = dilation).double())
                if batch_norm:
                    cn_block.append(nn.BatchNorm2d(out_ch))

                if activation == 'relu':
                    cn_block.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    cn_block.append(nn.LeakyReLU(0.1))

                for j in range(1, cn_in_block):
                    if stride != 1:
                        raise ValueError("Stride must be equal to 1 in larger blocks of cnns")
                    cn_block.append(nn.Conv2d(out_ch, out_ch, 3, stride,  dilation = dilation, padding='same').double())
                    if batch_norm:
                        cn_block.append(nn.BatchNorm2d(out_ch))

                    if activation == 'relu':
                        cn_block.append(nn.ReLU())
                    elif activation == 'leaky_relu':
                        cn_block.append(nn.LeakyReLU(0.1))


                convs.append(nn.Sequential(*cn_block))

            else:
                convs.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride,  dilation = dilation).double())
                if batch_norm:
                    convs.append(nn.BatchNorm2d(out_ch))

                if activation == 'relu':
                    convs.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    convs.append(nn.LeakyReLU(0.1))

            if upsample_layers > 0 and i < upsample_layers:
                in_ch = out_ch + up_out_chs[upsample_layers - i - 1]
            else:
                in_ch = out_ch

            pool_corr = 0
            if pooling == 'avg':
                convs.append(nn.AvgPool2d(2, stride=1))
                pool_corr = 1
            elif pooling == 'max':
                convs.append(nn.MaxPool2d(2, stride=1))
                pool_corr = 1

            if ratio_type == 'sqrt':
                fr = np.sqrt(fr)
            elif ratio_type == 'sq':
                fr = fr ** 2

            self.conv_out_dim = floor((self.conv_out_dim - dilation*(self.kernel_size -1) - 1)/stride + 1) - pool_corr

        self.convs = nn.ModuleList(convs)
        self.conv_num = conv_num
        self.out_ch = out_ch

        self.fc_num = fc_num
        self.fc_in_dim = self.out_ch * (self.conv_out_dim * self.conv_out_dim)

        if fc_trans == 'smooth':
            self.fc_dims = [int(self.fc_in_dim * ((self.fc_in_dim / fc_units) ** (-i/self.fc_num))) for i in range(1, self.fc_num + 1)]
        elif fc_trans == 'const':
            self.fc_dims = [fc_units for i in range(1, self.fc_num + 1)]

        self.d1 = nn.Linear(self.fc_in_dim, self.fc_dims[0])
        self.relu = nn.ReLU()

        fc_layers = []
        for i in range(0, fc_num - 1):
            fc_layers.append(nn.Linear(self.fc_dims[i], self.fc_dims[i+1]))
            if activation == 'relu':
                fc_layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                fc_layers.append(nn.LeakyReLU(0.1))

            if dropout:
                fc_layers.append(nn.Dropout())


        self.fc_layers = nn.Sequential(*fc_layers)
        self.d3 = nn.Linear(self.fc_dims[-1], output_size)


    def forward(self, x):
        if self.upconv_num > 0:
            xs = [x]
            for i, upconv in enumerate(self.upconvs):
                x = upconv(x)
                if i % 2 == 1:
                    xs.append(x)

        for i, conv in enumerate(self.convs):
            if i <= 1:
                x = conv(x)
            elif self.upconv_num > 0 and (i // 2) < (self.upconv_num + 1) and i % 2 == 0:
                x = conv(torch.cat((x, xs[self.upconv_num - (i // 2)]), dim=1))
            else:
                x = conv(x)


        x = x.view(-1, self.out_ch * (self.conv_out_dim * self.conv_out_dim))
        x = self.d1(x)
        x = self.relu(x)

        x = self.fc_layers(x)

        output = self.d3(x)

        if self.mode == 'classifier':
            output = torch.sigmoid(output)

        return output


class RecursiveCNN(nn.Module):
    def __init__(self, qbits_num, output_size, conv_num, fc_num, kernel_size, filters_ratio, dilation = 1, ratio_type = None, pooling = None, dropout = False, mode = 'regression', upsample_layers = 0, fc_trans = "const", fc_units = 128, input_channels = 2, stride = 1, activation = 'relu', tensor_layers = False, tensor_map = False, pretrained_model_path = None, device = "cpu"):
        super(RecursiveCNN, self).__init__()
        self.qbits_num = qbits_num
        self.device = device

        if qbits_num == 2:
            self.n_1model = CNN(qbits_num, output_size, conv_num, fc_num, kernel_size, filters_ratio, dilation, ratio_type, pooling, dropout, mode, upsample_layers, fc_trans, fc_units, input_channels, stride, activation, tensor_layers, tensor_map)
        else:
            self.n_1model = RecursiveCNN(qbits_num - 1, bipartitions_num(qbits_num - 1), conv_num, fc_num, kernel_size, filters_ratio, dilation, ratio_type, pooling, dropout, mode = 'regression', upsample_layers = upsample_layers, fc_trans = fc_trans, fc_units = fc_units, input_channels = input_channels, stride = stride, activation = activation, tensor_layers = tensor_layers, tensor_map = tensor_map)

        if pretrained_model_path != None:
            self.n_1model.load_state_dict(torch.load(pretrained_model_path))

        self.out_layer = nn.Linear(bipartitions_num(qbits_num - 1)*qbits_num, output_size)

    def forward(self, x):
        if self.qbits_num == 2:
            return self.n_1model(x)
        else:
            rhos = torch.complex(x[:,0,:,:], x[:,1,:,:]).cpu().numpy()
            torch_rhos_c = [torch.stack([torch.tensor(partial_trace(DensityMatrix(rho), [q]).data).to(self.device) for rho in rhos], dim=0) for q in range(self.qbits_num)]
            torch_rhos = [torch.stack((rho.real, rho.imag), dim = 1) for rho in torch_rhos_c]
            out = torch.cat([self.n_1model(rho) for rho in torch_rhos], dim = 1)
            return torch.sigmoid(self.out_layer(out))


# General parallel dilation model consisting of:
# - dilation*conv_num convolutional layers, each with kernel_size and number of output channels based on input channels and filters_ratio as well as ratio_type,
#       each such layer is followed by ReLU activation function
# - parallel conv_num dilation convolutional layers, which output is stacked together with output of classical convolutions
#       every i-th convloutional layer (i = dilation) to guarantee that dimensions match each other
# - fc_num fully connected layers, each consisting of 128 units and ReLU activation function,
#       to each such layer dropout can be added if it is set to True
# - finally the network can work both in regression and classifier mode with output_size
# - to make the model scalable for different number of qubits, qbits_num is also needed
class DCNN(nn.Module):
    def __init__(self, qbits_num, output_size, conv_num, fc_num, kernel_size, filters_ratio, dilation = 1, ratio_type = None, dropout = False, mode = 'regression'):
        super(DCNN, self).__init__()

        self.input_channels = 2
        self.dim = 2**qbits_num
        self.mode = mode

        convs = []
        dil_convs = []
        fr = filters_ratio
        in_ch = self.input_channels
        for i in range(0, conv_num):
            out_ch = int(in_ch*fr)
            dil_convs.append(nn.Conv2d(in_ch, out_ch, kernel_size, dilation= dilation).double())
            for j in range(0, dilation):
                convs.append(nn.Conv2d(in_ch, out_ch, kernel_size).double())
                in_ch = out_ch
            in_ch = 2*out_ch

            if ratio_type == 'sqrt':
                fr = np.sqrt(fr)
            elif ratio_type == 'sq':
                fr = fr ** 2

        self.convs = nn.ModuleList(convs)
        self.dil_convs = nn.ModuleList(dil_convs)
        self.conv_num = conv_num
        self.dilation = dilation

        self.kernel_size = kernel_size
        self.conv_out_dim = self.dim - self.conv_num*dilation*(self.kernel_size - 1)
        self.out_ch = 2*out_ch
        self.fc_num = fc_num

        fc_layers = []
        for i in range(0, fc_num - 1):
            fc_layers.append(nn.Linear(128, 128))
            fc_layers.append(nn.ReLU())

            if dropout:
                fc_layers.append(nn.Dropout())

        self.d1 = nn.Linear(self.out_ch * (self.conv_out_dim * self.conv_out_dim), 128)
        self.relu = nn.ReLU()
        self.fc_layers = nn.Sequential(*fc_layers)
        self.d3 = nn.Linear(128, output_size)


    def forward(self, x):
        for i in range(self.conv_num):
            x1 = self.dil_convs[i](x)
            x1 = F.relu(x1, inplace=True)
            x2 = x
            for j in range(self.dilation):
                x2 = self.convs[i*self.dilation + j](x2)
                x2 = F.relu(x2, inplace=True)
            x = torch.cat((x1, x2), dim= 1)

        x = x.view(-1, self.out_ch * (self.conv_out_dim * self.conv_out_dim))
        x = self.d1(x)
        x = self.relu(x)

        x = self.fc_layers(x)

        output = self.d3(x)

        if self.mode == 'classifier':
            output = torch.sigmoid(output)

        return output
