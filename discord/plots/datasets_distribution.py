import sys
sys.path.append('./')

from commons.data.datasets import BipartitionMatricesDataset
from commons.models.separators import FancySeparator, rho_reconstruction

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


separator_path = './models/3qbits/FancySeparator_l1_all_sep_o48_fc4_bl.pt'

val_dictionary_path = './datasets/3qbits/val_bisep_no_pptes/negativity_bipartitions.txt'
val_root_dir = './datasets/3qbits/val_bisep_no_pptes/matrices/'

discordant_dictionary_path = './datasets/3qbits/val_bisep_discordant/negativity_bipartitions.txt'
discordant_root_dir = './datasets/3qbits/val_bisep_discordant/matrices/'

pure_dictionary_path = './datasets/3qbits/pure_test/negativity_bipartitions.txt'
pure_root_dir = './datasets/3qbits/pure_test/matrices/'

mixed_dictionary_path = './datasets/3qbits/mixed_test/negativity_bipartitions.txt'
mixed_root_dir = './datasets/3qbits/mixed_test/matrices/'

acin_dictionary_path = './datasets/3qbits/acin_test/negativity_bipartitions.txt'
acin_root_dir = './datasets/3qbits/acin_test/matrices/'

horodecki_dictionary_path = './datasets/3qbits/horodecki_test/negativity_bipartitions.txt'
horodecki_root_dir = './datasets/3qbits/horodecki_test/matrices/'

bennet_dictionary_path = './datasets/3qbits/bennet_test/negativity_bipartitions.txt'
bennet_root_dir = './datasets/3qbits/bennet_test/matrices/'

datasets = [
    # ('Validation', val_dictionary_path, val_root_dir),
    ('Pure', pure_dictionary_path, pure_root_dir),
    ('Mixed', mixed_dictionary_path, mixed_root_dir),
    # ('Discordant', discordant_dictionary_path, discordant_root_dir),
    ('Acin et al.', acin_dictionary_path, acin_root_dir),
    ('Horodecki', horodecki_dictionary_path, horodecki_root_dir),
    ('UPB', bennet_dictionary_path, bennet_root_dir)
]

results_dir = './results/3qbits/multi_class/'
save_file_name = 'full_distrib_dens_log.png'

logscale = True

qbits_num = 3
sep_ch = 24
sep_fc_num = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

separator = FancySeparator(qbits_num, sep_ch, fc_layers = sep_fc_num)
separator.double()
separator.load_state_dict(torch.load(separator_path))
separator.eval()
separator.to(device)

criterion = nn.L1Loss(reduction='none')

if logscale:
    bins = np.geomspace(0.0001, 0.1, 30)
else:
    bins = np.linspace(0.0001, 0.1, 30)


def get_distribution(dataset_info, subset_size = 10000, class_id = 0):
    dataset = BipartitionMatricesDataset(dataset_info[1], dataset_info[2], 0.0001)
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=subset_size, shuffle=False)
    x, labels = next(iter(loader))
    x = x.to(device)
    separator_output = separator(x)
    rho = rho_reconstruction(x, separator_output)
    if type(class_id) == int:
        mask = torch.all(labels == class_id, dim=1)
    elif type(class_id) == list and len(class_id) == 2:
        mask = torch.logical_and(*[torch.any(labels == id, dim = 1) for id in class_id])
    else:
        raise ValueError('class_id must be either int or list of two ints')
    loss = criterion(rho[mask], x[mask]).view(-1, torch.prod(torch.tensor(rho.shape[1:]))).mean(1)
    return loss.detach().cpu().numpy()


def plot_distribution(dataset_info, distribution, states_type = 'separable'):
    p, x = np.histogram(distribution, bins=bins)
    p = p / np.sum(p)
    plt.plot(x[:-1], p, label=f'{dataset_info[0]} dataset: {states_type} states')
    plt.fill_between(x[:-1], p, alpha=0.3)


plt.figure(figsize=(10, 10))
# set font size
plt.rcParams.update({'font.size': 18})
if logscale:
    plt.xscale('log')

for dataset_info in datasets:
    distribution_0 = get_distribution(dataset_info, class_id=0)
    distribution_1 = get_distribution(dataset_info, class_id=1)
    distribution_01 = get_distribution(dataset_info, class_id=[0, 1])

    # plot histogram if the distributions are not empty
    if len(distribution_0) > 0:
        plot_distribution(dataset_info, distribution_0, states_type='separable')

    if len(distribution_1) > 0:
        plot_distribution(dataset_info, distribution_1, states_type='entangled')

    if len(distribution_01) > 0:
        plot_distribution(dataset_info, distribution_01, states_type='biseparable')
    
plt.legend()
plt.xlabel('L1 loss')
plt.ylabel('Density of samples')
plt.savefig(results_dir + save_file_name)
