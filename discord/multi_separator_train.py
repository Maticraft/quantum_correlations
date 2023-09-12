import sys
import os
sys.path.append('./')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from commons.data.datasets import DensityMatricesDataset
from commons.models.cnns import CNN
from commons.models.separators import FancySeparator, Separator, SiameseFancySeparator

from commons.pytorch_utils import save_acc
from commons.test_utils.base import test
from commons.test_utils.separator import test_separator, test_separator_as_classifier
from commons.train_utils.base import train
from commons.train_utils.separator import train_separator, train_siamese_separator


# Common params
data_dir = './datasets/3qbits/'
metrics = 'negativity'

batch_size = 128
batch_interval = 400
epochs = 10
learning_rate = 0.001
threshold = 0.001

qbits_num = 3
output_dim = 2
dilation = 1
kernel_size = 3
fr = 16
thresh = 0.0004
num_separators = 20
PREVIOUS_SEPARATOR_THRESHOLD = 0.001
out_channels_per_ratio = 24
ratio_type = 'sqrt'
pooling = 'None'
input_channels = 2 #if larger than 2, then noise is generated

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

gl_sep_val_set = DensityMatricesDataset(data_dir + 'val_separable/dictionary.txt', data_dir + 'val_separable/matrices', metrics, threshold)
gl_sep_val_loader = DataLoader(gl_sep_val_set, batch_size= batch_size, shuffle = True)

gl_mixed_bal_test_set = DensityMatricesDataset(data_dir + 'mixed_test_bal/dictionary.txt', data_dir + 'mixed_test_bal/matrices', "negativity", threshold)
gl_mixed_bal_test_loader = DataLoader(gl_mixed_bal_test_set, batch_size=batch_size)

gl_mixed_bal_test_disc_set = DensityMatricesDataset(data_dir + 'mixed_test_bal/dictionary.txt', data_dir + 'mixed_test_bal/matrices', "discord", threshold)
gl_mixed_bal_test_disc_loader = DataLoader(gl_mixed_bal_test_disc_set, batch_size=batch_size)

save_path_loss = './models/3qbits/multi_sep_th2/FancySeparator_l1_{}_o48_{}bl_it{}.pt'

train_path =  './results/3qbits/multi_sep_th2/l1_sep_{}_{}it{}_train_loss.txt'
thresholds_path = './results/3qbits/multi_sep_th2/l1_sep_{}_{}thresholds.txt'

params0 = {
    'data_name': 'all_sep',
    'train_dir': 'train_separable',
    'fc_name': 'fc4_',
    'fc': 4,
}
params01 = {
    'data_name': 'all_sep',
    'train_dir': 'train_separable',
    'fc_name': '',
    'fc': 0,
}

params_list = [params0, params01]

def perform_computations(params, previous_separators = [], prev_frq_of_excluded_states = 0., prev_separator_threshold = PREVIOUS_SEPARATOR_THRESHOLD):
    train_set = DensityMatricesDataset(data_dir + f'{params["train_dir"]}/dictionary.txt', data_dir + f'{params["train_dir"]}/matrices', metrics, threshold)
    train_loader = DataLoader(train_set, batch_size= batch_size, shuffle = True)

    model = FancySeparator(qbits_num, out_channels_per_ratio, input_channels, fc_layers=params['fc'])
    try:
        model.load_state_dict(torch.load(save_path_loss.format(params['data_name'], params['fc_name'], len(previous_separators))))
        print('Model loaded')
    except:
        print('Model not found')
    model.double()
    model.to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    best_loss = 1000.
    best_ep1 = 0
    threshold_increased = False

    criterion = nn.L1Loss(reduction='none')
    save_acc(train_path.format(params['data_name'], params['fc_name'], len(previous_separators)), "Epoch", ["train_loss", "validation loss", "frq_of_excluded_states"], write_mode='w')
    os.makedirs(os.path.dirname(save_path_loss.format(params['data_name'], params['fc_name'], len(previous_separators))), exist_ok=True)

    for epoch_number in range(1, epochs + 1):
        train_loss, frq_of_excluded_states = train_separator(model, device, train_loader, optimizer, criterion, epoch_number, batch_interval, use_noise=False, enforce_symmetry=False, previous_separators=previous_separators, previous_separators_threshold=prev_separator_threshold)

        if frq_of_excluded_states < 1.1 * prev_frq_of_excluded_states and not threshold_increased:
            prev_separator_threshold = 2 * prev_separator_threshold
            threshold_increased = True

        loss, _ = test_separator_as_classifier(model, device, gl_sep_val_loader, criterion, "Pure val set", prev_separator_threshold)

        if loss < best_loss:
            torch.save(model.state_dict(), save_path_loss.format(params['data_name'], params['fc_name'], len(previous_separators)))
            best_loss = loss
            best_ep1 = epoch_number

        save_acc(train_path.format(params['data_name'], params['fc_name'], len(previous_separators)), epoch_number, [train_loss, loss, frq_of_excluded_states])

    print("Best epoch loss: {}".format(best_ep1))
    model.load_state_dict(torch.load(save_path_loss.format(params['data_name'], params['fc_name'], len(previous_separators))))
    return model, frq_of_excluded_states, prev_separator_threshold


previous_separators = []
frq_of_excluded_states = 0.
prev_separator_threshold = PREVIOUS_SEPARATOR_THRESHOLD
save_acc(thresholds_path.format(params0['data_name'], params0['fc_name']), "Separator_it", ["Threshold", "Frq of excluded states"], write_mode='w')

for i in range(num_separators):
    print(i)
    separator, frq_of_excluded_states, prev_separator_threshold = perform_computations(params0, previous_separators, frq_of_excluded_states, prev_separator_threshold)
    previous_separators.append(separator)
    save_acc(thresholds_path.format(params0['data_name'], params0['fc_name']), i, [prev_separator_threshold, frq_of_excluded_states])
