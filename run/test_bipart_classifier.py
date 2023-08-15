import sys
from commons.data.datasets import BipartitionMatricesDataset
from commons.models.cnns import CNN

from commons.models.separator_classifiers import FancySeparatorEnsembleClassifier
from commons.models.separator_classifiers import FancyClassifier
from commons.test_utils.base import test
from commons.train_utils import train
sys.path.append('./')

import os

import torch
from torch.utils.data import DataLoader

from commons.pytorch_utils import save_acc

train_dictionary_path = './datasets/3qbits/train_bisep_no_pptes/negativity_bipartitions.txt'
train_root_dir = './datasets/3qbits/train_bisep_no_pptes/matrices/'

val_dictionary_path = './datasets/3qbits/val_bisep_no_pptes/negativity_bipartitions.txt'
val_root_dir = './datasets/3qbits/val_bisep_no_pptes/matrices/'

mixed_dictionary_path = './datasets/3qbits/mixed_test/negativity_bipartitions.txt'
mixed_root_dir = './datasets/3qbits/mixed_test/matrices/'

acin_dictionary_path = './datasets/3qbits/acin_test/negativity_bipartitions.txt'
acin_root_dir = './datasets/3qbits/acin_test/matrices/'

biseparable_dictionary_path = './datasets/3qbits/biseparable_test/negativity_bipartitions.txt'
biseparable_root_dir = './datasets/3qbits/biseparable_test/matrices/'

results_dir = './results/3qbits/nopptes_bisep_test/'
results_file = 'fancy_sep_ens10_class.txt'

model_dir = './models/3qbits/nopptes_bisep/'
model_name = 'fancy_sep_ens10_class'

batch_size = 128
batch_interval = 800
qbits_num = 3
sep_ch = 16
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

test_biseparable_dataset = BipartitionMatricesDataset(biseparable_dictionary_path, biseparable_root_dir, 0.0001)
test_biseparable_loader = DataLoader(test_biseparable_dataset, batch_size=batch_size, shuffle=True)

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

model_path = model_dir + model_name + '.pt'
results_path = results_dir + results_file

model = FancySeparatorEnsembleClassifier(qbits_num, sep_ch, sep_fc_num, train_dataset.bipart_num, 10)
# model = FancyClassifier(qbits_num, sep_ch, sep_fc_num, 5, train_dataset.bipart_num, 128)
# model = CNN(qbits_num, train_dataset.bipart_num, 3, 5, 2, 16, ratio_type='sqrt', mode='classifier')
model.double()
model.load_state_dict(torch.load(model_path))
criterion = torch.nn.BCELoss()

save_acc(results_path, '', ['Train loss', 'Train_acc', 'Validation loss', 'Validation accuracy', 'Mixed loss', 'Mixed accuracy', 'ACIN loss', 'ACIN accuracy', 'Bisep loss', 'Bisep accuracy'], write_mode='w')

train_loss, train_acc = test(model, device, train_loader, criterion, "Train data set", bipart=True)
val_loss, val_acc = test(model, device, val_loader, criterion, "Validation data set", bipart=True)
mixed_loss, mixed_acc = test(model, device, test_mixed_loader, criterion, "Mixed data set", bipart=True)
acin_loss, acin_acc = test(model, device, test_acin_loader, criterion, "ACIN data set", bipart=True)    
biseparable_loss, biseparable_acc = test(model, device, test_biseparable_loader, criterion, "Biseparable data set", bipart=True)
save_acc(results_path, '', [train_loss, train_acc, val_loss, val_acc, mixed_loss, mixed_acc, acin_loss, acin_acc, biseparable_loss, biseparable_acc])
