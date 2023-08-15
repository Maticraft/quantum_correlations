import torch.nn.functional as F
from commons.models.cnns import CNN
from commons.models.separators import FancySeparator, Separator
from commons.pytorch_utils import rho_reconstruction


import torch
import torch.nn as nn


class FancySeparatorEnsembleClassifier(nn.Module):
    def __init__(self, qbits_num, sep_ch, sep_fc_num, output_size, ensemble_size):
        super(FancySeparatorEnsembleClassifier, self).__init__()
        self.ensemble_size = ensemble_size
        self.separators = nn.ModuleList([FancySeparator(qbits_num, sep_ch, fc_layers=sep_fc_num) for _ in range(ensemble_size)])
        self.classifier = CNN(qbits_num, output_size, 3, 5, 2, filters_ratio=4, ratio_type='sqrt', mode='classifier', input_channels=2*ensemble_size)

    def forward(self, x):
        separators_outputs = [separator(x) for separator in self.separators]
        reconstructed_matrices = [rho_reconstruction(x, separator_output) for separator_output in separators_outputs]
        reconstructed_matrices = torch.stack(reconstructed_matrices, dim = 1)
        difference_matrices = reconstructed_matrices - x.unsqueeze(1)
        difference_matrices = difference_matrices.view(-1, 2*self.ensemble_size, x.shape[-2], x.shape[-1])
        classifier_output = self.classifier(difference_matrices)
        return classifier_output


class SeparatorCNN(nn.Module):
    def __init__(self, qbits_num, sep_out_ch_per_rho, sep_kernel_size, output_size, convs_num, cnn_kernel_size = 3, fc_num = 5, filters_ratio = 16, ratio_type = 'sqrt', mode = 'classifier', include_original_matrix = False):
        super(SeparatorCNN, self).__init__()
        self.Separator = Separator(qbits_num, sep_out_ch_per_rho, filters_ratio, sep_kernel_size, ratio_type=ratio_type)
        self.include_original_matrix = include_original_matrix
        if self.include_original_matrix:
            self.CNN = CNN(qbits_num, output_size, convs_num, fc_num, cnn_kernel_size, filters_ratio, ratio_type=ratio_type, mode=mode, input_channels=4)
        else:
            self.CNN = CNN(qbits_num, output_size, convs_num, fc_num, cnn_kernel_size, filters_ratio, ratio_type=ratio_type, mode=mode)

    def forward(self, x):
        x_prim = self.Separator(x)
        x_diff = torch.abs(x - rho_reconstruction(x, x_prim))
        if self.include_original_matrix:
            output = self.CNN(torch.cat((x, x_diff), dim = 1))
        else:
            output = self.CNN(x_diff)
        return output


# Entanglement classifier based on input from Separator
class SeparatorClassifier(nn.Module):
    def __init__(self, qbits_num, channels_per_rho, filters_ratio, output_size, fc_num):
        super(SeparatorClassifier, self).__init__()

        self.dim = 2
        self.input_ch = qbits_num*2*channels_per_rho
        self.out_ch = int(self.input_ch*filters_ratio)

        self.conv = nn.Conv2d(self.input_ch, self.out_ch, 2)

        self.fc_num = fc_num
        self.fc_dims = [int(self.out_ch * ((self.out_ch / 128) ** (-i/self.fc_num))) for i in range(1, self.fc_num + 1)]

        self.f1 = nn.Linear(self.out_ch, self.fc_dims[0])

        fc_layers = []
        for i in range(0, fc_num - 1):
            fc_layers.append(nn.Linear(self.fc_dims[i], self.fc_dims[i+1]))
            fc_layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*fc_layers)

        self.fout = nn.Linear(self.fc_dims[-1] + 1, output_size)


    def forward(self, x, sep_met):
        x = F.relu(self.conv(x))
        x = x.view(-1, self.out_ch)
        x = F.relu(self.f1(x))
        x = self.fc_layers(x)

        x = torch.cat((x, sep_met), dim=1)
        output = torch.sigmoid(self.fout(x))

        return output


class CombinedClassifier(nn.Module):
    def __init__(self, qbits_num, output_size, conv_num, fc_num, kernel_size, filters_ratio, dilation = 1, ratio_type = None, pooling = None, dropout = False, mode = 'regression', upsample_layers = 0, fc_trans = "const", fc_units = 128, stride = 1, activation = 'relu', tensor_layers = False, tensor_map = False, sep_ch = 32):
        super(CombinedClassifier, self).__init__()

        self.CNN = CNN(qbits_num, output_size, conv_num, fc_num, kernel_size, filters_ratio, dilation, ratio_type, pooling, dropout, mode, upsample_layers, fc_trans, fc_units, 4, stride, activation, tensor_layers, tensor_map)
        self.separator = FancySeparator(qbits_num, sep_ch)
        self.dim = self.CNN.dim

    def forward(self, x):
        sep_matrices = self.separator(x)
        rho = rho_reconstruction(x, sep_matrices)
        new_data = torch.cat((x, rho), dim = 1)
        output = self.CNN(new_data)
        return output