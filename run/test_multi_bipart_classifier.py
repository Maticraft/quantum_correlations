import sys
sys.path.append('./')

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from commons.data.datasets import BipartitionMatricesDataset
from commons.models.cnns import CNN

from commons.models.separator_classifiers import FancySeparatorEnsembleClassifier
from commons.models.separator_classifiers import FancyClassifier
from commons.models.siamese_networks import VectorSiamese
from commons.models.separators import FancySeparator
from commons.test_utils.multi_classifier import test_multi_classifier
from commons.pytorch_utils import save_acc

train_dictionary_path = './datasets/3qbits/train_bisep_no_pptes/negativity_bipartitions.txt'
train_root_dir = './datasets/3qbits/train_bisep_no_pptes/matrices/'

val_dictionary_path = './datasets/3qbits/val_bisep_no_pptes/negativity_bipartitions.txt'
val_root_dir = './datasets/3qbits/val_bisep_no_pptes/matrices/'

mixed_dictionary_path = './datasets/3qbits/mixed_test/negativity_bipartitions.txt'
mixed_root_dir = './datasets/3qbits/mixed_test/matrices/'

acin_dictionary_path = './datasets/3qbits/acin_test/negativity_bipartitions.txt'
acin_root_dir = './datasets/3qbits/acin_test/matrices/'

pseudo_acin_dictionary_path = './datasets/3qbits/pseudo_acin_test/negativity_bipartitions.txt'
pseudo_acin_root_dir = './datasets/3qbits/pseudo_acin_test/matrices/'

pseudo_acin_pptes_dictionary_path = './datasets/3qbits/pseudo_acin_pptes_test/negativity_bipartitions.txt'
pseudo_acin_pptes_root_dir = './datasets/3qbits/pseudo_acin_pptes_test/matrices/'

pseudo_acin_no_pptes_dictionary_path = './datasets/3qbits/test_no_pptes_acin/negativity_bipartitions.txt'
pseudo_acin_no_pptes_root_dir = './datasets/3qbits/test_no_pptes_acin/matrices/'

biseparable_dictionary_path = './datasets/3qbits/biseparable_test/negativity_bipartitions.txt'
biseparable_root_dir = './datasets/3qbits/biseparable_test/matrices/'

separator_path = './models/3qbits/FancySeparator_l1_all_sep_o48_fc4_bl.pt'

model_dir = './models/3qbits/multi_class/nopptes_bisep/'
model_name = 'siam_cnn_class_{}'

results_dir = './results/3qbits/multi_class/nopptes_bisep/'
results_file = 'siam_class.txt'

thresholds = [0., 5.e-4, 1.e-3, 2.e-3, 5.e-3, 1.e-2, 2.e-2, 5.e-2]
# thresholds = np.geomspace(0.0001, 0.1, 15)
# thresholds = np.insert(thresholds, 0, 0.)

batch_size = 128
batch_interval = 800
qbits_num = 3
sep_ch = 24
sep_fc_num = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = BipartitionMatricesDataset(train_dictionary_path, train_root_dir, 0.0001)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = BipartitionMatricesDataset(val_dictionary_path, val_root_dir, 0.0001)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_mixed_dataset = BipartitionMatricesDataset(mixed_dictionary_path, mixed_root_dir, 0.0001)
test_mixed_loader = DataLoader(test_mixed_dataset, batch_size=batch_size, shuffle=True)

test_acin_dataset = BipartitionMatricesDataset(acin_dictionary_path, acin_root_dir, 0.0001)
test_acin_loader = DataLoader(test_acin_dataset, batch_size=batch_size, shuffle=True)

test_pseudo_acin_dataset = BipartitionMatricesDataset(pseudo_acin_dictionary_path, pseudo_acin_root_dir, 0.0001)
test_pseudo_acin_loader = DataLoader(test_pseudo_acin_dataset, batch_size=batch_size, shuffle=True)

test_pseudo_acin_pptes_dataset = BipartitionMatricesDataset(pseudo_acin_pptes_dictionary_path, pseudo_acin_pptes_root_dir, 0.0001)
test_pseudo_acin_pptes_loader = DataLoader(test_pseudo_acin_pptes_dataset, batch_size=batch_size, shuffle=True)

test_pseudo_acin_no_pptes_dataset = BipartitionMatricesDataset(pseudo_acin_no_pptes_dictionary_path, pseudo_acin_no_pptes_root_dir, 0.0001)
test_pseudo_acin_no_pptes_loader = DataLoader(test_pseudo_acin_no_pptes_dataset, batch_size=batch_size, shuffle=True)

test_biseparable_dataset = BipartitionMatricesDataset(biseparable_dictionary_path, biseparable_root_dir, 0.0001)
test_biseparable_loader = DataLoader(test_biseparable_dataset, batch_size=batch_size, shuffle=True)

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

criterion = torch.nn.BCELoss()

separator = FancySeparator(qbits_num, sep_ch, fc_layers = sep_fc_num)
separator.double()
separator.load_state_dict(torch.load(separator_path))
separator.eval()

model_paths = [model_dir + model_name.format(i) + '.pt' for i in range(len(thresholds) - 1)]
results_path = results_dir + results_file

models = []
for model_path in model_paths:
    # model = FancySeparatorEnsembleClassifier(qbits_num, sep_ch, sep_fc_num, train_dataset.bipart_num, 3)
    # model = FancyClassifier(qbits_num, sep_ch, sep_fc_num, 5, train_dataset.bipart_num, 128)
    # model = CNN(qbits_num, train_dataset.bipart_num, 3, 5, 2, 16, ratio_type='sqrt', mode='classifier')
    model = VectorSiamese(qbits_num, train_dataset.bipart_num, 3, 5, 2, 16, ratio_type='sqrt', mode='classifier', biparts_mode='all')
    model.double()
    model.load_state_dict(torch.load(model_path))
    models.append(model)
print('Models loaded')

save_acc(results_path, '', ['Train loss', 'Train_acc', 'Validation loss', 'Validation accuracy', 'Mixed loss', 'Mixed accuracy', 'ACIN loss', 'ACIN accuracy', 'Pseudo ACIN loss', 'Pseudo ACIN accuracy',  'Pseudo ACIN pptes loss', 'Pseudo ACIN pptes accuracy', 'Pseudo ACIN no pptes loss', 'Pseudo ACIN no pptes accuracy', 'Bisep loss', 'Bisep accuracy'], write_mode='w')

train_loss, train_acc = test_multi_classifier(models, separator, thresholds, device, train_loader, criterion, "Train data set", bipart=True)
val_loss, val_acc = test_multi_classifier(models, separator, thresholds, device, val_loader, criterion, "Validation data set", bipart=True)
mixed_loss, mixed_acc = test_multi_classifier(models, separator, thresholds, device, test_mixed_loader, criterion, "Mixed data set", bipart=True)
acin_loss, acin_acc = test_multi_classifier(models, separator, thresholds, device, test_acin_loader, criterion, "ACIN data set", bipart=True)    
pseudo_acin_loss, pseudo_acin_acc = test_multi_classifier(models, separator, thresholds, device, test_pseudo_acin_loader, criterion, "Pseudo ACIN data set", bipart=True)
pseudo_acin_pptes_loss, pseudo_acin_pptes_acc = test_multi_classifier(models, separator, thresholds, device, test_pseudo_acin_pptes_loader, criterion, "Pseudo ACIN pptes data set", bipart=True)
pseudo_acin_no_pptes_loss, pseudo_acin_no_pptes_acc = test_multi_classifier(models, separator, thresholds, device, test_pseudo_acin_no_pptes_loader, criterion, "Pseudo ACIN no pptes data set", bipart=True)
biseparable_loss, biseparable_acc = test_multi_classifier(models, separator, thresholds, device, test_biseparable_loader, criterion, "Biseparable data set", bipart=True)
save_acc(results_path, '', [train_loss, train_acc, val_loss, val_acc, mixed_loss, mixed_acc, acin_loss, acin_acc, pseudo_acin_loss, pseudo_acin_acc,\
                            pseudo_acin_pptes_loss, pseudo_acin_pptes_acc, pseudo_acin_no_pptes_loss, pseudo_acin_no_pptes_acc, biseparable_loss, biseparable_acc])
