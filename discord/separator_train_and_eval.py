import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from commons.models.cnns import CNN
from commons.models.separators import FancySeparator, Separator, SiameseFancySeparator

from commons.pytorch_utils import train, test, save_acc, CombinedSeparator, train_separator, train_siamese_separator, test_separator, test_separator_as_classifier


class DensityMatricesDataset(Dataset):

    def __init__(self, dictionary, root_dir, metrics, threshold, data_limit = None):
        self.dictionary = self.load_dict(dictionary)
        self.root_dir = root_dir
        self.metrics = metrics
        self.threshold = threshold
        self.data_limit = data_limit

        if self.metrics == "global_entanglement":
            self.label_pos = 3

        elif self.metrics == "von_Neumann":
            self.label_pos = 4

        elif self.metrics == "concurrence":
            self.label_pos = 5

        elif self.metrics == "negativity":
            self.label_pos = 6
        
        elif self.metrics == "realignment":
            self.label_pos = 7
            
        elif self.metrics == "discord":
            self.label_pos = 8

        elif self.metrics == "trace":
            self.label_pos = 9

        else:
            raise ValueError('Wrong metrics')
      
    def __len__(self):
        if self.data_limit != None:
            return self.data_limit
        else:
            return len(self.dictionary)


    def __getitem__(self, idx):
      if torch.is_tensor(idx):
        idx = idx.tolist()

      matrix_name = os.path.join(self.root_dir, self.dictionary[idx][0] + ".npy")
      matrix = np.load(matrix_name)
      matrix_r = np.real(matrix)
      matrix_im = np.imag(matrix)

      tensor = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0))

      label = float(self.dictionary[idx][self.label_pos])
      if label > self.threshold:
        label = 1
      else:
        label = 0
      label = torch.tensor(label).double()
      label = label.unsqueeze(0)

      return (tensor, label)


    def load_dict(self, filepath):
      
      with open(filepath, 'r') as dictionary:
        data = dictionary.readlines()

      parsed_data = [row.rstrip("\n").split(', ') for row in data]

      return parsed_data


# Common params
data_dir = './datasets/'
metrics = 'negativity'

batch_size = 128
batch_interval = 400
epochs = 20
learning_rate = 0.001
threshold = 0.001

qbits_num = 3
output_dim = 2
dilation = 1
kernel_size = 3
fr = 16
thresh = 0.0004
out_channels_per_ratio = 24
ratio_type = 'sqrt'
pooling = 'None'
input_channels = 2 #if larger than 2, then noise is generated

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

gl_sep_val_set = DensityMatricesDataset(data_dir + 'val_separable/dictionary.txt', data_dir + 'val_separable/matrices', metrics, threshold)
gl_sep_val_loader = DataLoader(gl_sep_val_set, batch_size= batch_size, shuffle = True)

gl_mixed_bal_test_set = DensityMatricesDataset(data_dir + 'mixed_test_balanced_disc/dictionary.txt', data_dir + 'mixed_test_balanced_disc/matrices', "negativity", threshold)
gl_mixed_bal_test_loader = DataLoader(gl_mixed_bal_test_set, batch_size=batch_size)

gl_mixed_bal_test_disc_set = DensityMatricesDataset(data_dir + 'mixed_test_balanced_disc/dictionary.txt', data_dir + 'mixed_test_balanced_disc/matrices', "discord", threshold)
gl_mixed_bal_test_disc_loader = DataLoader(gl_mixed_bal_test_disc_set, batch_size=batch_size)

save_path_acc = './classifiers/FancySeparator_l1_{}_3q_o48_{}bacc.pt'
save_path_loss = './classifiers/FancySeparator_l1_{}_3q_o48_{}bl.pt'

count_save_path = './results/discord/bures_sep_{}_{}prediction_thresh_mixed_bal_bal_acc_log.txt'
train_path =  './results/discord/bures_sep_{}_{}train_loss.txt'

params0 = {
    'data_name': 'all_sep',
    'train_dir': 'train_sep',
    'fc_name': 'fc4_',
    'fc': 4,
}
params01 = {
    'data_name': 'all_sep',
    'train_dir': 'train_sep',
    'fc_name': '',
    'fc': 0,
}
params1 = {
    'data_name': 'pure_sep_param',
    'train_dir': 'train_pure_separable',
    'fc_name': '',
    'fc': 0,
}
params2 = {
    'data_name': 'prod',
    'train_dir': 'train_product',
    'fc_name': '',
    'fc': 0,
}
params3 = {
    'data_name': 'prod',
    'train_dir': 'train_product',
    'fc_name': 'fc4_',
    'fc': 4,
}
params4 = {
    'data_name': 'zd',
    'train_dir': 'train_zd',
    'fc_name': '',
    'fc': 0,
}
params5 = {
    'data_name': 'zd',
    'train_dir': 'train_zd',
    'fc_name': 'fc4_',
    'fc': 4,
}
params6 = {
    'data_name': 'nps',
    'train_dir': 'train_non_product',
    'fc_name': '',
    'fc': 0,
}
params7 = {
    'data_name': 'nps',
    'train_dir': 'train_non_product',
    'fc_name': 'fc4_',
    'fc': 4,
}


params_list = [params4, params5, params6, params7]

def perform_computations(params):
    train_set = DensityMatricesDataset(data_dir + f'{params["train_dir"]}/dictionary.txt', data_dir + f'{params["train_dir"]}/matrices', metrics, threshold)
    train_loader = DataLoader(train_set, batch_size= batch_size, shuffle = True)

    model = FancySeparator(qbits_num, out_channels_per_ratio, input_channels, fc_layers=params['fc'])
    try:
        model.load_state_dict(torch.load(save_path_loss.format(params['data_name'], params['fc_name'])))
        print('Model loaded')
    except:
        print('Model not found')
    model.double()
    model.to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    best_acc = 0
    best_loss = 1000.
    best_ep1 = 0
    best_ep2 = 0
    thresh = 0.0001

    criterion = nn.L1Loss(reduction='none')
    save_acc(train_path.format(params['data_name'], params['fc_name']), "Epoch", ["train_loss", "validation loss"])

    for epoch_number in range(1, epochs + 1):
        train_loss = train_separator(model, device, train_loader, optimizer, 'bures', epoch_number, batch_interval, use_noise=False, enforce_symmetry=False)

        loss, acc = test_separator_as_classifier(model, device, gl_sep_val_loader, 'bures', "Pure val set", thresh)

        if loss < best_loss:
            torch.save(model.state_dict(), save_path_loss.format(params['data_name'], params['fc_name']))
            best_loss = loss
            best_ep1 = epoch_number

        if acc > best_acc:
            torch.save(model.state_dict(), save_path_acc.format(params['data_name'], params['fc_name']))
            best_acc = acc
            best_ep2 = epoch_number

        save_acc(train_path.format(params['data_name'], params['fc_name']), epoch_number, [train_loss, loss])

    print("Best epoch loss: {}".format(best_ep1))
    print("Best epoch acc: {}".format(best_ep2))

    model.load_state_dict(torch.load(save_path_loss.format(params['data_name'], params['fc_name'])))

    thresholds = np.geomspace(0.0001, 1., 100)

    criterion = nn.L1Loss(reduction='none')

    save_acc(count_save_path.format(params['data_name'], params['fc_name']), "Threshold", accuracies=["Pred/ZeroDisc", "Prec_ZeroDisc", "Recall_ZeroDisc", "Pred/Separable", "Prec_Separable", "Recall_Separable", "BACC_ZeroDisc", "BACC_Separable"])

    for th in thresholds:
        print("Threshold = {}".format(th))

        l_ent, acc_ent, cm_ent = test_separator_as_classifier(model, device, gl_mixed_bal_test_loader, 'bures', "MEnt:", th, use_noise=False, confusion_matrix=True)
        l_disc, acc_disc, cm_disc = test_separator_as_classifier(model, device, gl_mixed_bal_test_disc_loader, 'bures', "MDisc:", th, use_noise=False, confusion_matrix=True)

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
            count_save_path.format(params['data_name'], params['fc_name']),
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

for i, params in enumerate(params_list):
    print(i)
    perform_computations(params)

