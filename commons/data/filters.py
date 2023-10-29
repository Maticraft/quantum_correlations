from commons.models.separators import FancySeparator, rho_reconstruction

import torch
import torch.nn as nn


def separator_filter(x, separator, thresholds_range, criterion = nn.L1Loss(reduction='mean')):
    if x.shape[0] != 1:
        x = x.unsqueeze(0)
    separator.to(x.device)
    separator_output = separator(x)
    rho = rho_reconstruction(x, separator_output)
    loss = criterion(rho, x).item()
    if thresholds_range[0] < loss < thresholds_range[1]:
        return True
    return False


def target_filter(y, target):
    return torch.all(y == torch.tensor(target)).item()


def filter_data_with_separator_and_target(data, separator, thresholds_range, target):
    x, y = data
    if separator_filter(x, separator, thresholds_range) and target_filter(y, target):
        return True
    return False


def init_default_separator_filter():
    separator_path = './models/3qbits/FancySeparator_l1_all_sep_o48_fc4_bl.pt'
    qbits_num = 3
    sep_ch = 24
    sep_fc_num = 4
    separator = FancySeparator(qbits_num, sep_ch, fc_layers = sep_fc_num)
    separator.double()
    separator.load_state_dict(torch.load(separator_path))
    separator.eval()
    
    def default_separator_filter(data, thresholds_range):
        return separator_filter(data, separator, thresholds_range)
    
    return default_separator_filter
