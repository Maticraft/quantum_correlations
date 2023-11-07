import sys
sys.path.append('./')

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from commons.data.datasets import BipartitionMatricesDataset, DataFilteredSubset
from commons.data.filters import separator_filter
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

horodecki_dictionary_path = './datasets/3qbits/horodecki_test/negativity_bipartitions.txt'
horodecki_root_dir = './datasets/3qbits/horodecki_test/matrices/'

bennet_dictionary_path = './datasets/3qbits/bennet_test/negativity_bipartitions.txt'
bennet_root_dir = './datasets/3qbits/bennet_test/matrices/'

biseparable_dictionary_path = './datasets/3qbits/biseparable_test/negativity_bipartitions.txt'
biseparable_root_dir = './datasets/3qbits/biseparable_test/matrices/'

separator_path = './models/3qbits/FancySeparator_l1_all_sep_o48_fc4_bl.pt'

model_dir = './models/3qbits/multi_class_siam/nopptes_bisep/'
model_name = 'weights05_ep10_cnn_class_best_val_loss_{}'

results_dir = './results/3qbits/multi_class_siam/nopptes_bisep/'
results_file = 'weights05_ep10_cnn_class_best_val_subsets.txt'

thresholds = [0., 5.e-4, 1.e-3, 2.e-3, 5.e-3, 1.e-2, 2.e-2, 5.e-2]
# thresholds = [0., 1.e-3, 1.e-1]
# thresholds = np.geomspace(0.0001, 0.1, 15)
# thresholds = np.insert(thresholds, 0, 0.)
# thresholds = np.geomspace(0.001, 0.1, 20)
# thresholds = np.insert(thresholds, 0, 0.)
# thresholds = np.delete(thresholds, len(thresholds) - 1)

batch_size = 128
batch_interval = 800
qbits_num = 3
sep_ch = 24
sep_fc_num = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

separator = FancySeparator(qbits_num, sep_ch, fc_layers = sep_fc_num)
separator.double()
separator.load_state_dict(torch.load(separator_path))
separator.eval()

train_dataset = BipartitionMatricesDataset(train_dictionary_path, train_root_dir, 0.0001)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = BipartitionMatricesDataset(val_dictionary_path, val_root_dir, 0.0001)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print('Filtering data...')

threshold_range = [0.001, 0.01]
val_subset = DataFilteredSubset(val_dataset, lambda x: separator_filter(x, separator, threshold_range))
val_subset_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

threshold_range2 = [0.01, 0.1]
val_subset_subset2 = DataFilteredSubset(val_dataset, lambda x: separator_filter(x, separator, threshold_range2))
val_subset_subset2_loader = DataLoader(val_subset_subset2, batch_size=batch_size, shuffle=True)

test_mixed_dataset = BipartitionMatricesDataset(mixed_dictionary_path, mixed_root_dir, 0.0001)
test_mixed_loader = DataLoader(test_mixed_dataset, batch_size=batch_size, shuffle=True)

test_mixed_subset = DataFilteredSubset(test_mixed_dataset, lambda x: separator_filter(x, separator, threshold_range))
test_mixed_subset_loader = DataLoader(test_mixed_subset, batch_size=batch_size, shuffle=True)

test_mixed_subset2 = DataFilteredSubset(test_mixed_dataset, lambda x: separator_filter(x, separator, threshold_range2))
test_mixed_subset2_loader = DataLoader(test_mixed_subset2, batch_size=batch_size, shuffle=True)

print('Data filtered')

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

criterion = torch.nn.BCELoss()

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

save_acc(results_path, '', ['Train loss', 'Train_acc', 'Validation loss', 'Validation accuracy', f'Validation ({threshold_range}) loss', f'Validation ({threshold_range}) accuracy', f'Validation ({threshold_range2}) loss', f'Validation ({threshold_range2}) accuracy',
                            'Mixed loss', 'Mixed accuracy', f'Mixed ({threshold_range}) loss',f'Mixed ({threshold_range}) accuracy', f'Mixed ({threshold_range2}) loss', f'Mixed ({threshold_range2}) accuracy'], write_mode='w')

train_loss, train_acc, train_cm  = test_multi_classifier(models, separator, thresholds, device, train_loader, criterion, "Train data set", bipart=True, confusion_matrix=True, confusion_matrix_dim=2)
val_loss, val_acc, val_cm = test_multi_classifier(models, separator, thresholds, device, val_loader, criterion, "Validation data set", bipart=True, confusion_matrix=True, confusion_matrix_dim=2)
val_s_loss, val_s_acc, val_s_cm  = test_multi_classifier(models, separator, thresholds, device, val_subset_loader, criterion, f"Validation data set data set range: {threshold_range}", bipart=True, confusion_matrix=True, confusion_matrix_dim=2)
val_s2_loss, val_s2_acc, val_s2_cm  = test_multi_classifier(models, separator, thresholds, device, val_subset_subset2_loader, criterion, f"Validation data set data set range: {threshold_range2}", bipart=True, confusion_matrix=True, confusion_matrix_dim=2)
mixed_loss, mixed_acc, mixed_cm = test_multi_classifier(models, separator, thresholds, device, test_mixed_loader, criterion, "Mixed data set", bipart=True, confusion_matrix=True, confusion_matrix_dim=2)
mixed_s_loss, mixed_s_acc, mixed_s_cm = test_multi_classifier(models, separator, thresholds, device, test_mixed_subset_loader, criterion, f"Mixed data set range: {threshold_range}", bipart=True, confusion_matrix=True, confusion_matrix_dim=2)
mixed_s2_loss, mixed_s2_acc, mixed_s2_cm = test_multi_classifier(models, separator, thresholds, device, test_mixed_subset2_loader, criterion, f"Mixed data set range: {threshold_range2}", bipart=True, confusion_matrix=True, confusion_matrix_dim=2)

save_acc(results_path, '', [train_loss, train_acc, val_loss, val_acc, val_s_loss, val_s_acc, val_s2_loss, val_s2_acc,
                            mixed_loss, mixed_acc, mixed_s_loss, mixed_s_acc, mixed_s2_loss, mixed_s2_acc])
