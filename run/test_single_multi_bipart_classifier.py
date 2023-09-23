import sys
sys.path.append('./')

from commons.data.datasets import BipartitionMatricesDataset, FilteredSubset
from commons.models.cnns import CNN

from commons.models.separator_classifiers import FancySeparatorEnsembleClassifier
from commons.models.separator_classifiers import FancyClassifier
from commons.models.separators import FancySeparator, separator_filter
from commons.test_utils.base import test
from commons.train_utils.base import train
from commons.pytorch_utils import save_acc

import os

import numpy as np
import torch
from torch.utils.data import DataLoader


train_dictionary_path = './datasets/3qbits/train_bisep_no_pptes/negativity_bipartitions.txt'
train_root_dir = './datasets/3qbits/train_bisep_no_pptes/matrices/'

separator_path = './models/3qbits/FancySeparator_l1_all_sep_o48_fc4_bl.pt'

val_dictionary_path = './datasets/3qbits/val_bisep_no_pptes/negativity_bipartitions.txt'
val_root_dir = './datasets/3qbits/val_bisep_no_pptes/matrices/'

mixed_dictionary_path = './datasets/3qbits/mixed_test/negativity_bipartitions.txt'
mixed_root_dir = './datasets/3qbits/mixed_test/matrices/'

acin_dictionary_path = './datasets/3qbits/acin_test/negativity_bipartitions.txt'
acin_root_dir = './datasets/3qbits/acin_test/matrices/'

results_dir = './results/3qbits/multi_class/nopptes_bisep/'
model_dir = './models/3qbits/multi_class/nopptes_bisep/'
model_name = 'cnn_class_3'

threshold_range = [2.e-3, 5.e-3]

batch_size = 128
batch_interval = 800
qbits_num = 3
sep_ch = 24
sep_fc_num = 4
epoch_num = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = BipartitionMatricesDataset(train_dictionary_path, train_root_dir, 0.0001)
val_dataset = BipartitionMatricesDataset(val_dictionary_path, val_root_dir, 0.0001)
test_mixed_dataset = BipartitionMatricesDataset(mixed_dictionary_path, mixed_root_dir, 0.0001)
test_acin_dataset = BipartitionMatricesDataset(acin_dictionary_path, acin_root_dir, 0.0001)

separator = FancySeparator(qbits_num, sep_ch, fc_layers = sep_fc_num)
separator.double()
separator.load_state_dict(torch.load(separator_path))
separator.eval()

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

model_path = model_dir + model_name + '.pt'
results_path = results_dir + model_name + '_test.txt'

save_acc(results_path, '', ['Validation loss', 'Validation accuracy', 'Mixed loss', 'Mixed accuracy', 'ACIN loss', 'ACIN accuracy', f'Threshold range: {str(threshold_range)}'], write_mode='w')

eval_flags = {'val': True, 'mixed': True, 'acin': True}

try:
    val_subset = FilteredSubset(val_dataset, lambda x: separator_filter(x, separator, threshold_range))
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
except:
    eval_flags['val'] = False

try:
    test_mixed_subset = FilteredSubset(test_mixed_dataset, lambda x: separator_filter(x, separator, threshold_range))
    test_mixed_loader = DataLoader(test_mixed_subset, batch_size=batch_size, shuffle=True)
except:
    eval_flags['mixed'] = False

try:
    test_acin_subset = FilteredSubset(test_acin_dataset, lambda x: separator_filter(x, separator, threshold_range))
    test_acin_loader = DataLoader(test_acin_subset, batch_size=batch_size, shuffle=True)
except:
    eval_flags['acin'] = False

# model = FancySeparatorEnsembleClassifier(qbits_num, sep_ch, sep_fc_num, train_dataset.bipart_num, 3)
# model = FancyClassifier(qbits_num, sep_ch, sep_fc_num, 5, train_dataset.bipart_num, 128)
model = CNN(qbits_num, train_dataset.bipart_num, 3, 5, 2, 16, ratio_type='sqrt', mode='classifier')
model.double()
model.load_state_dict(torch.load(model_path))

criterion = torch.nn.BCELoss()

val_loss, val_acc = None, None
mixed_loss, mixed_acc = None, None
acin_loss, acin_acc = None, None

if eval_flags['val']:
    val_loss, val_acc, val_cm = test(model, device, val_loader, criterion, "Validation data set", bipart=True, confusion_matrix=True)
if eval_flags['mixed']:
    mixed_loss, mixed_acc, mixed_cm = test(model, device, test_mixed_loader, criterion, "Mixed data set", bipart=True, confusion_matrix=True)
if eval_flags['acin']:
    acin_loss, acin_acc, acin_cm = test(model, device, test_acin_loader, criterion, "ACIN data set", bipart=True, confusion_matrix=True)    
save_acc(results_path, '', [val_loss, val_acc, mixed_loss, mixed_acc, acin_loss, acin_acc])
