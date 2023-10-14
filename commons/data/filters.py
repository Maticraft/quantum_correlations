from commons.models.separators import rho_reconstruction

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
