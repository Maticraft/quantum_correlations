import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from commons.data.datasets import DensityMatricesDataset

from commons.pytorch_utils import save_acc
from commons.trace import test_trace_predictions


# Common params
data_dir = './datasets/'
batch_size = 128
threshold = 1.e-3

gl_mixed_bal_test_set = DensityMatricesDataset(data_dir + 'mixed_test_balanced_disc/dictionary.txt', data_dir + 'mixed_test_balanced_disc/matrices', "negativity", threshold)
gl_mixed_bal_test_loader = DataLoader(gl_mixed_bal_test_set, batch_size=batch_size)

gl_mixed_bal_test_disc_set = DensityMatricesDataset(data_dir + 'mixed_test_balanced_disc/dictionary.txt', data_dir + 'mixed_test_balanced_disc/matrices', "discord", threshold)
gl_mixed_bal_test_disc_loader = DataLoader(gl_mixed_bal_test_disc_set, batch_size=batch_size)

count_save_path = './results/discord/bures_trace_prediction_thresh_mixed_bal_bal_acc_log_low.txt'

def perform_computations():
    thresholds = np.geomspace(1.e-8, 0.0001, 100)

    save_acc(count_save_path, "Threshold", accuracies=["Pred/ZeroDisc", "Prec_ZeroDisc", "Recall_ZeroDisc", "Pred/Separable", "Prec_Separable", "Recall_Separable", "BACC_ZeroDisc", "BACC_Separable"])

    for th in thresholds:
        print("Threshold = {}".format(th))

        l_ent, acc_ent, cm_ent = test_trace_predictions(gl_mixed_bal_test_loader, 'bures', th, "MEnt:", confusion_matrix=True)
        l_disc, acc_disc, cm_disc = test_trace_predictions(gl_mixed_bal_test_disc_loader, 'bures', th, "MDisc:", confusion_matrix=True)

        zdapz = cm_disc[0, 0]
        zd = cm_disc[0, 0] + cm_disc[0, 1]
        sapz = cm_ent[0, 0]
        sep = cm_ent[0, 0] + cm_ent[0, 1]
        pred_zero = cm_ent[0, 0] + cm_ent[1, 0]
        if (cm_ent[0, 0] + cm_ent[0, 1] == 0) or (cm_ent[1, 0] + cm_ent[1, 1] == 0):
            bal_acc_ent = acc_ent
        else:
            bal_acc_ent = .5* cm_ent[0, 0] / (cm_ent[0, 0] + cm_ent[0, 1]) + .5 * cm_ent[1, 1] / (cm_ent[1, 0] + cm_ent[1, 1])

        if (cm_disc[0, 0] + cm_disc[0, 1] == 0) or (cm_disc[1, 0] + cm_disc[1, 1] == 0):
            bal_acc_disc = acc_disc
        else:
            bal_acc_disc = .5* cm_disc[0, 0] / (cm_disc[0, 0] + cm_disc[0, 1]) + .5 * cm_disc[1, 1] / (cm_disc[1, 0] + cm_disc[1, 1])

        save_acc(
            count_save_path,
            th,
            [   
                zdapz / (zd + pred_zero - zdapz + 1.e-7),
                zdapz/(pred_zero + 1.e-7),
                zdapz/(zd + 1.e-7),
                sapz / (sep + pred_zero - sapz + 1.e-7),
                sapz / (pred_zero + 1.e-7), 
                sapz / (sep + 1.e-7),
                bal_acc_disc,
                bal_acc_ent
            ]
        )

perform_computations()
