import sys
sys.path.append('./')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import DataLoader

from commons.data.datasets import BipartitionMatricesDataset
from commons.models.cnns import CNN
from commons.pytorch_utils import load_acc, count_parameters
from commons.test_utils.base import test


model_name = 'cnn_class_best_val_paper'
model_dir = './paper_models/3qbits/nopptes_bisep/arch_scaling_valid/'

val_dictionary_path = './datasets/3qbits/val_bisep_no_pptes/negativity_bipartitions.txt'
val_root_dir = './datasets/3qbits/val_bisep_no_pptes/matrices/'

batch_size = 128
dataset = BipartitionMatricesDataset(val_dictionary_path, val_root_dir, 0.0001)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fc_layers = [1, 3, 5, 1, 3, 5, 1, 3, 5]
conv_layers = [1, 1, 1, 2, 2, 2, 3, 3, 3]
filters_ratios = [2, 4, 8, 16]

criterion = torch.nn.BCELoss()

params_number = {}
accuracies = {}
losses = {}
for fcl_num, conv_num in zip(fc_layers, conv_layers):
    params_number[f'{fcl_num}fc_{conv_num}conv'] = []
    accuracies[f'{fcl_num}fc_{conv_num}conv'] = []
    losses[f'{fcl_num}fc_{conv_num}conv'] = []
    for filter_ratio in filters_ratios:
        model = CNN(3, 3, conv_num, fcl_num, 2, filter_ratio, ratio_type='sqrt', mode='classifier')
        num_params = count_parameters(model)
        model_name_i = f'{model_name}_{fcl_num}fc_{conv_num}conv_{filter_ratio}filt_{num_params}params'
        model_path = model_dir + model_name_i + '.pt'
        model.double()
        model.load_state_dict(torch.load(model_path))

        loss, acc = test(model, device, data_loader, criterion, "Validation data set", bipart=True)
        params_number[f'{fcl_num}fc_{conv_num}conv'].append(num_params)
        accuracies[f'{fcl_num}fc_{conv_num}conv'].append(np.mean(acc))
        losses[f'{fcl_num}fc_{conv_num}conv'].append(loss)


plt.rcParams.update({'font.size': 28})
plt.rcParams.update({'lines.markersize': 10})

fig = plt.figure(figsize=(10, 10))
# plt.title('Accuracy vs number of parameters')
plt.plot(params_number['1fc_1conv'], accuracies['1fc_1conv'], 'o', c='r', markerfacecolor='none', label='1 fc layer, 1 conv layer')
plt.plot(params_number['3fc_1conv'], accuracies['3fc_1conv'], 'o', c='g', markerfacecolor='none', label='3 fc layers, 1 conv layer')
plt.plot(params_number['5fc_1conv'], accuracies['5fc_1conv'], 'o', c='b', markerfacecolor='none', label='5 fc layers, 1 conv layer')
plt.plot(params_number['1fc_2conv'], accuracies['1fc_2conv'], 'x', c='r', markerfacecolor='none', label='1 fc layer, 2 conv layers')
plt.plot(params_number['3fc_2conv'], accuracies['3fc_2conv'], 'x', c='g', markerfacecolor='none', label='3 fc layers, 2 conv layers')
plt.plot(params_number['5fc_2conv'], accuracies['5fc_2conv'], 'x', c='b', markerfacecolor='none', label='5 fc layers, 2 conv layers')
plt.plot(params_number['1fc_3conv'], accuracies['1fc_3conv'], 's', c='r', markerfacecolor='none', label='1 fc layer, 3 conv layers')
plt.plot(params_number['3fc_3conv'], accuracies['3fc_3conv'], 's', c='g', markerfacecolor='none', label='3 fc layers, 3 conv layers')
plt.plot(params_number['5fc_3conv'], accuracies['5fc_3conv'], 's', c='b', markerfacecolor='none', label='5 fc layers, 3 conv layers')
plt.xlabel('Number of model parameters')
plt.ylabel('Accuracy [%]')
plt.xscale('log')
plt.xticks([1e4, 1e5, 1e6])
# add custom separate legends:
# - for colors: red - 1 fc layer, green - 3 fc layers, blue - 5 fc layers
red_patch = mpatches.Patch(color='red', label='1 fc layer')
green_patch = mpatches.Patch(color='green', label='3 fc layers')
blue_patch = mpatches.Patch(color='blue', label='5 fc layers')
# - for markers: o - 1 conv layer, x - 2 conv layers, s - 3 conv layers
o_handle = plt.Line2D([], [], color='black', marker='o', markerfacecolor='none', linestyle='None', markersize=15, label='1 conv layer')
x_handle = plt.Line2D([], [], color='black', marker='x', markerfacecolor='none', linestyle='None', markersize=15, label='2 conv layers')
s_handle = plt.Line2D([], [], color='black', marker='s', markerfacecolor='none', linestyle='None', markersize=15, label='3 conv layers')
lgd = fig.legend(handles=[red_patch, green_patch, blue_patch, o_handle, x_handle, s_handle], bbox_to_anchor = (1.35, 0.94))
plt.tight_layout(pad = 2)
plt.savefig('./plots/parameters_dependance_acc.png', bbox_extra_artists=[lgd], pad_inches = 0.3, bbox_inches='tight')
plt.close()


fig = plt.figure(figsize=(10, 10))
# plt.title('Loss vs number of parameters')
plt.plot(params_number['1fc_1conv'], losses['1fc_1conv'], 'o', markerfacecolor='none', c='r', label='1 fc layer, 1 conv layer')
plt.plot(params_number['3fc_1conv'], losses['3fc_1conv'], 'o', markerfacecolor='none', c='g', label='3 fc layers, 1 conv layer')
plt.plot(params_number['5fc_1conv'], losses['5fc_1conv'], 'o', markerfacecolor='none', c='b', label='5 fc layers, 1 conv layer')
plt.plot(params_number['1fc_2conv'], losses['1fc_2conv'], 'x', markerfacecolor='none', c='r', label='1 fc layer, 2 conv layers')
plt.plot(params_number['3fc_2conv'], losses['3fc_2conv'], 'x', markerfacecolor='none', c='g', label='3 fc layers, 2 conv layers')
plt.plot(params_number['5fc_2conv'], losses['5fc_2conv'], 'x', markerfacecolor='none', c='b', label='5 fc layers, 2 conv layers')
plt.plot(params_number['1fc_3conv'], losses['1fc_3conv'], 's', markerfacecolor='none', c='r', label='1 fc layer, 3 conv layers')
plt.plot(params_number['3fc_3conv'], losses['3fc_3conv'], 's', markerfacecolor='none', c='g', label='3 fc layers, 3 conv layers')
plt.plot(params_number['5fc_3conv'], losses['5fc_3conv'], 's', markerfacecolor='none', c='b', label='5 fc layers, 3 conv layers')
plt.xlabel('Number of model parameters')
plt.ylabel('Loss')
plt.xscale('log')
# set xtivks to 10^4, 10^5, 10^6
plt.xticks([1e4, 1e5, 1e6])
# add custom separate legends:
# - for colors: red - 1 fc layer, green - 3 fc layers, blue - 5 fc layers
red_patch = mpatches.Patch(color='red', label='1 fc layer')
green_patch = mpatches.Patch(color='green', label='3 fc layers')
blue_patch = mpatches.Patch(color='blue', label='5 fc layers')
# - for markers: o - 1 conv layer, x - 2 conv layers, s - 3 conv layers
o_handle = plt.Line2D([], [], color='black', marker='o', markerfacecolor='none', linestyle='None', markersize=15, label='1 conv layer')
x_handle = plt.Line2D([], [], color='black', marker='x', markerfacecolor='none', linestyle='None', markersize=15, label='2 conv layers')
s_handle = plt.Line2D([], [], color='black', marker='s', markerfacecolor='none', linestyle='None', markersize=15, label='3 conv layers')
lgd = fig.legend(handles=[red_patch, green_patch, blue_patch, o_handle, x_handle, s_handle], bbox_to_anchor = (1.35, 0.94))
plt.tight_layout(pad = 2)
plt.savefig('./plots/parameters_dependance_loss.png', bbox_extra_artists=[lgd], pad_inches = 0.3, bbox_inches='tight')
plt.close()
