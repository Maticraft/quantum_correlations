import sys
sys.path.append('./')

from commons.data.datasets import BipartitionMatricesDataset, FilteredSubset
from commons.data.filters import filter_data_with_separator_and_target
from commons.models.cnns import CNN
from commons.models.separators import FancySeparator
from commons.models.separator_classifiers import FancySeparatorEnsembleClassifier
from commons.models.separator_classifiers import FancyClassifier
from commons.test_utils.base import test
from commons.train_utils.base import train

import os

import torch
from torch.utils.data import DataLoader

from commons.pytorch_utils import save_acc

separator_path = './models/3qbits/FancySeparator_l1_all_sep_o48_fc4_bl.pt'

train_dictionary_path = './datasets/3qbits/train_bisep_no_pptes/negativity_bipartitions.txt'
train_root_dir = './datasets/3qbits/train_bisep_no_pptes/matrices/'

val_dictionary_path = './datasets/3qbits/val_bisep_discordant/negativity_bipartitions.txt'
val_root_dir = './datasets/3qbits/val_bisep_discordant/matrices/'

mixed_dictionary_path = './datasets/3qbits/mixed_test/negativity_bipartitions.txt'
mixed_root_dir = './datasets/3qbits/mixed_test/matrices/'

acin_dictionary_path = './datasets/3qbits/acin_test/negativity_bipartitions.txt'
acin_root_dir = './datasets/3qbits/acin_test/matrices/'

horodecki_dictionary_path = f'./datasets/3qbits/horodecki_test/negativity_bipartitions.txt'
horodecki_root_dir = f'./datasets/3qbits/horodecki_test/matrices/'

bennet_dictionary_path = f'./datasets/3qbits/bennet_test/negativity_bipartitions.txt'
bennet_root_dir = f'./datasets/3qbits/bennet_test/matrices/'


results_dir = './results/3qbits/nopptes_bisep_filtered/'
model_dir = './models/3qbits/nopptes_bisep_filtered/'
model_name = 'cnn_class'

batch_size = 128
batch_interval = 800
qbits_num = 3
sep_ch = 24
sep_fc_num = 4
epoch_num = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

separator = FancySeparator(qbits_num, sep_ch, fc_layers = sep_fc_num)
separator.double()
separator.load_state_dict(torch.load(separator_path))
separator.eval()

threshold_range = (2.e-3, 1.e-1)
target = 0

print('Filtering data...')
train_dataset = BipartitionMatricesDataset(train_dictionary_path, train_root_dir, 0.0001, format='npy', filename_pos=0)
train_subset = FilteredSubset(train_dataset, lambda x: not filter_data_with_separator_and_target(x, separator, threshold_range, target))
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

val_dataset = BipartitionMatricesDataset(val_dictionary_path, val_root_dir, 0.0001)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_mixed_dataset = BipartitionMatricesDataset(mixed_dictionary_path, mixed_root_dir, 0.0001)
test_mixed_loader = DataLoader(test_mixed_dataset, batch_size=batch_size, shuffle=True)

test_acin_dataset = BipartitionMatricesDataset(acin_dictionary_path, acin_root_dir, 0.0001)
test_acin_loader = DataLoader(test_acin_dataset, batch_size=batch_size, shuffle=True)

test_horodecki_dataset = BipartitionMatricesDataset(horodecki_dictionary_path, horodecki_root_dir, 0.0001)
test_horodecki_loader = DataLoader(test_horodecki_dataset, batch_size=batch_size, shuffle=True)

test_bennet_dataset = BipartitionMatricesDataset(bennet_dictionary_path, bennet_root_dir, 0.0001)
test_bennet_loader = DataLoader(test_bennet_dataset, batch_size=batch_size, shuffle=True)

# model = FancySeparatorEnsembleClassifier(qbits_num, sep_ch, sep_fc_num, train_dataset.bipart_num, 3)
# model = FancyClassifier(qbits_num, sep_ch, sep_fc_num, 5, train_dataset.bipart_num, 128)
model = CNN(qbits_num, train_dataset.bipart_num, 3, 5, 2, 16, ratio_type='sqrt', mode='classifier')
model.double()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

model_path = model_dir + model_name + '.pt'
results_path = results_dir + model_name + '.txt'

print('Training model...')
save_acc(results_path, 'Epoch', ['Train loss', 'Validation loss', 'Validation accuracy', 'Mixed loss', 'Mixed accuracy', 'ACIN loss', 'ACIN accuracy', 'Horodecki loss', 'Horodecki accuracy', 'Bennet loss', 'Bennet accuracy'], write_mode='w')

for epoch in range(epoch_num):
    train_loss = train(model, device, train_loader, optimizer, criterion, epoch, batch_interval, target_to_filter=-1)
    val_loss, val_acc = test(model, device, val_loader, criterion, "Validation data set", bipart=True)
    mixed_loss, mixed_acc = test(model, device, test_mixed_loader, criterion, "Mixed data set", bipart=True)
    acin_loss, acin_acc = test(model, device, test_acin_loader, criterion, "ACIN data set", bipart=True)    
    horodecki_loss, horodecki_acc = test(model, device, test_horodecki_loader, criterion, "Horodecki data set", bipart=True)
    bennet_loss, bennet_acc = test(model, device, test_bennet_loader, criterion, "Bennet data set", bipart=True)   
    save_acc(results_path, epoch, [train_loss, val_loss, val_acc, mixed_loss, mixed_acc, acin_loss, acin_acc, horodecki_loss, horodecki_acc,\
                            bennet_loss, bennet_acc])
    torch.save(model.state_dict(), model_path)
