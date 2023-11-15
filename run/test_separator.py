import sys
sys.path.append('./')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from commons.data.datasets import DensityMatricesDataset
from commons.models.separators import FancySeparator

from commons.pytorch_utils import save_acc
from commons.test_utils.base import test
from commons.test_utils.separator import test_separator, test_separator_as_classifier
from commons.train_utils.base import train
from commons.train_utils.separator import train_separator, train_siamese_separator


# Common params
data_dir = './datasets/3qbits/'
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

gl_mixed_bal_test_set = DensityMatricesDataset(data_dir + 'mixed_test_bal/dictionary.txt', data_dir + 'mixed_test_bal/matrices', "negativity", threshold)
gl_mixed_bal_test_loader = DataLoader(gl_mixed_bal_test_set, batch_size=batch_size)

gl_mixed_bal_test_disc_set = DensityMatricesDataset(data_dir + 'mixed_test_bal/dictionary.txt', data_dir + 'mixed_test_bal/matrices', "discord", threshold)
gl_mixed_bal_test_disc_loader = DataLoader(gl_mixed_bal_test_disc_set, batch_size=batch_size)

save_path_loss = './paper_models/3qbits/FancySeparator_l1_{}_o48_{}bl.pt'
count_save_path_ent = './results/3qbits/discord/l1_sep_{}_{}ent_prediction_thresh_mixed_bal_bal_acc_log.txt'
count_save_path_disc = './results/3qbits/discord/l1_sep_{}_{}disc_prediction_thresh_mixed_bal_bal_acc_log.txt'


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

def perform_computations(params):
    model = FancySeparator(qbits_num, out_channels_per_ratio, input_channels, fc_layers=params['fc'])
    model.load_state_dict(torch.load(save_path_loss.format(params['data_name'], params['fc_name'])))
    print('Model loaded')
    model.double()
    model.to(device)

    thresholds = np.geomspace(0.0001, 1., 100)

    criterion = nn.L1Loss(reduction='none')

    save_acc(count_save_path_ent.format(params['data_name'], params['fc_name']), "Threshold", accuracies=["Pred/Separable", "Prec_Separable", "Recall_Separable", "Pred/Entangled", "Prec_Entangled", "Recall_Entangled", "BACC"])
    save_acc(count_save_path_disc.format(params['data_name'], params['fc_name']), "Threshold", accuracies=["Pred/ZeroDiscord", "Prec_ZeroDiscord", "Recall_ZeroDiscord", "Pred/Discord", "Prec_Discord", "Recall_Discord", "BACC"])

    for th in thresholds:
        print("Threshold = {}".format(th))

        l_ent, acc_ent, cm_ent = test_separator_as_classifier(model, device, gl_mixed_bal_test_loader, criterion, "MEnt:", th, use_noise=False, confusion_matrix=True)

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
            count_save_path_ent.format(params['data_name'], params['fc_name']),
            th,
            [   
                sapz / (sep + pred_zero - sapz + 1.e-7),
                sapz / (pred_zero + 1.e-7), 
                sapz / (sep + 1.e-7),
                eapo / (ent + pred_one - eapo + 1.e-7),
                eapo / (pred_one + 1.e-7), 
                eapo / (ent + 1.e-7),
                bal_acc_ent
            ]
        )

        l_disc, acc_disc, cm_disc = test_separator_as_classifier(model, device, gl_mixed_bal_test_disc_loader, criterion, "MDisc:", th, use_noise=False, confusion_matrix=True)

        zdapz = cm_disc[0, 0]
        zd = cm_disc[0, 0] + cm_disc[0, 1]
        pred_zero = cm_disc[0, 0] + cm_disc[1, 0]

        dapz = cm_disc[1, 1]
        d = cm_disc[1, 0] + cm_disc[1, 1]
        pred_one = cm_disc[0, 1] + cm_disc[1, 1]

        if (cm_disc[0, 0] + cm_disc[0, 1] == 0) or (cm_disc[1, 0] + cm_disc[1, 1] == 0):
            bal_acc_disc = acc_disc
        else:
            bal_acc_disc = .5* cm_disc[0, 0] / (cm_disc[0, 0] + cm_disc[0, 1]) + .5 * cm_disc[1, 1] / (cm_disc[1, 0] + cm_disc[1, 1])

        save_acc(
            count_save_path_disc.format(params['data_name'], params['fc_name']),
            th,
            [
                zdapz / (zd + pred_zero - zdapz + 1.e-7),
                zdapz / (pred_zero + 1.e-7),
                zdapz / (zd + 1.e-7),
                dapz / (d + pred_one - dapz + 1.e-7),
                dapz / (pred_one + 1.e-7),
                dapz / (d + 1.e-7),
                bal_acc_disc
            ]
        )

for i, params in enumerate(params_list):
    print(i)
    perform_computations(params)
