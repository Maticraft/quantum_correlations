from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from commons.data.datasets import BipartitionMatricesDataset
from commons.models.separators import FancySeparator
from commons.test_utils.separator import separator_predict

num_qubits = 3
data_path = f'./datasets/3qbits/train_bisep_no_pptes/'
model_path = './paper_models/3qbits/FancySeparator_l1_all_sep_o48_fc4_bl.pt'

batch_size = 128
batch_interval = 800
qbits_num = 3
sep_ch = 24
sep_fc_num = 4
epoch_num = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FancySeparator(qbits_num, sep_ch, fc_layers = sep_fc_num)
model.double()
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

dataset = BipartitionMatricesDataset(data_path + 'negativity_bipartitions.txt', data_path + 'matrices', 0.001, return_metric_value=True)
dataloader = DataLoader(dataset, batch_size= batch_size, shuffle = True)

# Generate train data set
indx = 0

with open(data_path + f'bisep_distribution.txt', 'w') as f:
    for data, (target, metric) in tqdm(dataloader, desc='Labeling data'):
        data, target = data.to(device), target.to(device)     
        pred, loss = separator_predict(model, device, data, 0.1, return_loss=True)
        for loss_i, target_i, metric_i in zip(loss, target, metric):
            metric_avg = metric_i.mean().item()
            target_i_0 = torch.all(target_i == 0).item()
            target_i_1 = torch.all(target_i == 1).item()
            target_i = 0 if target_i_0 else 1
            target_i = 2 if not target_i_1 and not target_i_0 else target_i
            f.write(f'dens{indx}.npy, {loss_i.item()}, {target_i}, {metric_avg}\n')
            indx += 1
