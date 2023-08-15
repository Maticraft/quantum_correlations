import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from commons.data.datasets import DensityMatricesDataset
from commons.models.cnns import CNN
from commons.models.separators import FancySeparator, Separator, SiameseFancySeparator

from commons.pytorch_utils import save_acc
from commons.test.base import test
from commons.test.separator import test_separator, test_separator_as_classifier
from commons.train.base import train
from commons.train.separator import train_separator, train_siamese_separator


# Common params
data_dir = './data/3qbits/'
metrics = 'negativity'

batch_size = 128
threshold = 0.001

qbits_num = 3
out_channels_per_ratio = 24
ratio_type = 'sqrt'
pooling = 'None'
input_channels = 2 #if larger than 2, then noise is generated

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

gl_sep_val_set = DensityMatricesDataset(data_dir + 'val_separable/dictionary.txt', data_dir + 'val_separable/matrices', metrics, threshold)
gl_sep_val_loader = DataLoader(gl_sep_val_set, batch_size= batch_size, shuffle = True)

gl_pptes_hor = DensityMatricesDataset(data_dir + 'pptes_horodecki/dictionary.txt', data_dir + 'pptes_horodecki/matrices', "negativity", threshold)
gl_pptes_hor_loader = DataLoader(gl_pptes_hor, batch_size=batch_size)

gl_pptes_acin = DensityMatricesDataset(data_dir + 'pptes_acin/dictionary.txt', data_dir + 'pptes_acin/matrices', "negativity", threshold)
gl_pptes_acin_loader = DataLoader(gl_pptes_acin, batch_size=batch_size)

gl_pptes_bennet = DensityMatricesDataset(data_dir + 'pptes_bennet/dictionary.txt', data_dir + 'pptes_bennet/matrices', "negativity", threshold)
gl_pptes_bennet_loader = DataLoader(gl_pptes_bennet, batch_size=batch_size)

gl_pptes_2xd = DensityMatricesDataset(data_dir + 'pptes_2xd/dictionary.txt', data_dir + 'pptes_2xd/matrices', "negativity", threshold)
gl_pptes_2xd_loader = DataLoader(gl_pptes_2xd, batch_size=batch_size)

save_path_loss = './models/FancySeparator_l1_all_sep_3q_o48_fc4_bl.pt'

count_save_path = './results/l1_separator_{}_prediction_thresh_bal_acc_log.txt'

params0 = {
    'data_type': 'separable',
    'data_loader': gl_sep_val_loader
}
params1 = {
    'data_type': 'horodecki',
    'data_loader': gl_pptes_hor_loader
}
params2 = {
    'data_type': '2xd',
    'data_loader': gl_pptes_2xd_loader
}
params3 = {
    'data_type': 'acin',
    'data_loader': gl_pptes_acin_loader
}
params4 = {
    'data_type': 'bennet',
    'data_loader': gl_pptes_bennet_loader
}

params_list = [params0, params1, params2, params3, params4]

def perform_computations(params):
    model = FancySeparator(qbits_num, out_channels_per_ratio, input_channels, fc_layers=4)
    model.load_state_dict(torch.load(save_path_loss))
    print('Model loaded')
    model.double()
    model.to(device)

    thresholds = np.geomspace(0.0001, 1., 100)

    criterion = nn.L1Loss(reduction='none')

    save_acc(count_save_path.format(params['data_type']), "Threshold", accuracies=["Pred/Separable", "Prec_Separable", "Recall_Separable", "Pred/Entangled", "Prec_Entangled", "Recall_Entangled", "BACC"])

    for th in thresholds:
        print("Threshold = {}".format(th))

        l_ent, acc_ent, cm_ent = test_separator_as_classifier(model, device, params['data_loader'], criterion, "MEnt:", th, use_noise=False, confusion_matrix=True)

        sapz = cm_ent[0, 0]
        sep = cm_ent[0, 0] + cm_ent[0, 1]
        pred_zero = cm_ent[0, 0] + cm_ent[1, 0]

        eapo = cm_ent[1, 1]
        ent = cm_ent[1, 0] + cm_ent[1, 1]
        pred_one = cm_ent[0, 1] + cm_ent[1, 1]

        if (cm_ent[0, 0] + cm_ent[0, 1] == 0) or (cm_ent[1, 0] + cm_ent[1, 1] == 0):
            bal_acc_ent = acc_ent
        else:
            bal_acc_ent = .5* cm_ent[0, 0] / (cm_ent[0, 0] + cm_ent[0, 1]) + .5 * cm_ent[1, 1] / (cm_ent[1, 0] + cm_ent[1, 1])

        save_acc(
            count_save_path.format(params['data_type']),
            th,
            [   
                sapz / (sep + pred_zero - sapz + 1.e-7),
                sapz / (pred_zero + 1.e-7), 
                sapz / (sep + 1.e-7),
                eapo / (ent + pred_one - eapo + 1.e-7),
                eapo / (pred_one + 1.e-7), 
                eapo / (sep + 1.e-7),
                bal_acc_ent
            ]
        )

for i, params in enumerate(params_list):
    print(i)
    perform_computations(params)

