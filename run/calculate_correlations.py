import sys
sys.path.append('./')

from tqdm import tqdm
import os

import numpy as np
import torch
from qiskit.quantum_info import state_fidelity

from commons.data.datasets import DensityMatrixLoader
from commons.test_utils.separator import separator_predict
from commons.models.separator_classifiers import FancySeparator
from commons.pytorch_utils import save_acc

data_path = './datasets/3qbits/train_separable'
save_file_name = 'correlations.txt'
save_path_loss = './models/3qbits/FancySeparator_l1_all_sep_o48_fc4_bl.pt'

save_path = os.path.join(data_path, save_file_name)

matrix_loader = DensityMatrixLoader(data_path)

subsample_size = 1000
subsample1 = np.random.choice(len(matrix_loader), subsample_size, replace=False)
subsample2 = np.random.choice(len(matrix_loader), subsample_size, replace=False)

# Separator params
batch_size = 128
threshold = 0.001

qbits_num = 3
out_channels_per_ratio = 24
ratio_type = 'sqrt'
pooling = 'None'
input_channels = 2 #if larger than 2, then noise is generated

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FancySeparator(qbits_num, out_channels_per_ratio, input_channels, fc_layers=4)
model.load_state_dict(torch.load(save_path_loss))
print('Model loaded')
model.double()
model.to(device)


save_acc(save_path, 'Max loss', ['Loss diff', 'L1', 'Bures'], write_mode='w')
for idx1 in tqdm(subsample1, desc='Calculating statistics'):
    for idx2 in subsample2:
        matrix1, label1 = matrix_loader[idx1]
        matrix2, label2 = matrix_loader[idx2]
        fidelity = state_fidelity(matrix1, matrix2)
        bures = 2 * (1 - np.sqrt(fidelity))
        l2 = np.linalg.norm(matrix1.data - matrix2.data)

        matrix1_t = torch.from_numpy(matrix1.data)
        matrix1_t = torch.stack([matrix1_t.real, matrix1_t.imag], dim=0).unsqueeze(0)
        pred1, loss1 = separator_predict(model, device, matrix1_t, threshold, criterion='L1', return_loss=True)

        matrix2_t = torch.from_numpy(matrix2.data)
        matrix2_t = torch.stack([matrix2_t.real, matrix2_t.imag], dim=0).unsqueeze(0)
        pred2, loss2 = separator_predict(model, device, matrix2_t, threshold, criterion='L1', return_loss=True)

        loss_diff = torch.abs(loss1 - loss2).item()
        max_loss = torch.maximum(loss1, loss2).item()

        save_acc(save_path, max_loss, [loss_diff, l2, bures], write_mode='a')
