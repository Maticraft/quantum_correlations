import sys
sys.path.append('./')

import os

import torch
from torch.utils.data import DataLoader

from commons.data.datasets import BipartitionMatricesDataset
from commons.models.cnns import CNN
from commons.models.ensemble import Ensemble

from commons.models.separator_classifiers import FancySeparatorEnsembleClassifier
from commons.models.separator_classifiers import FancyClassifier
from commons.test_utils.base import test
from commons.train_utils.ensemble import train_ensemble
from commons.pytorch_utils import save_acc

batch_size = 128
batch_interval = 800
qbits_num = 3
epoch_num = 20

train_dictionary_path = f'./datasets/{qbits_num}qbits/train_bisep_discordant/negativity_bipartitions.txt'
train_root_dir = f'./datasets/{qbits_num}qbits/train_bisep_discordant/matrices/'

val_dictionary_path = f'./datasets/{qbits_num}qbits/val_bisep_discordant/negativity_bipartitions.txt'
val_root_dir = f'./datasets/{qbits_num}qbits/val_bisep_discordant/matrices/'

mixed_dictionary_path = f'./datasets/{qbits_num}qbits/mixed_test/negativity_bipartitions.txt'
mixed_root_dir = f'./datasets/{qbits_num}qbits/mixed_test/matrices/'

acin_dictionary_path = f'./datasets/{qbits_num}qbits/acin_test/negativity_bipartitions.txt'
acin_root_dir = f'./datasets/{qbits_num}qbits/acin_test/matrices/'

horodecki_dictionary_path = f'./datasets/{qbits_num}qbits/horodecki_test/negativity_bipartitions.txt'
horodecki_root_dir = f'./datasets/{qbits_num}qbits/horodecki_test/matrices/'

bennet_dictionary_path = f'./datasets/{qbits_num}qbits/bennet_test/negativity_bipartitions.txt'
bennet_root_dir = f'./datasets/{qbits_num}qbits/bennet_test/matrices/'

biseparable_dictionary_path = f'./datasets/{qbits_num}qbits/biseparable_test/negativity_bipartitions.txt'
biseparable_root_dir = f'./datasets/{qbits_num}qbits/biseparable_test/matrices/'

results_dir = f'./results/{qbits_num}qbits/nopptes_bisep/'
model_dir = f'./models/{qbits_num}qbits/nopptes_bisep/'
model_name = 'ensemble_cnn_class'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = BipartitionMatricesDataset(train_dictionary_path, train_root_dir, 0.0001)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

test_biseparable_dataset = BipartitionMatricesDataset(biseparable_dictionary_path, biseparable_root_dir, 0.0001)
test_biseparable_loader = DataLoader(test_biseparable_dataset, batch_size=batch_size, shuffle=True)

ensemble_size = 6

cnn_params = {
    'qbits_num': qbits_num,
    'output_size': train_dataset.bipart_num,
    'conv_num': 3,
    'fc_num': 5,
    'kernel_size': 2,
    'filters_ratio': 16,
    'dilation': 1,
    'ratio_type': 'sqrt',
    'mode': 'classifier'
}

selector_cnn_params = {
    'qbits_num': qbits_num,
    'output_size': ensemble_size,
    'conv_num': 3,
    'fc_num': 5,
    'kernel_size': 2,
    'filters_ratio': 16,
    'dilation': 1,
    'ratio_type': 'sqrt',
    'mode': 'regression'
}

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

model_path = model_dir + model_name + '.pt'
results_path = results_dir + model_name + '.txt'

model = Ensemble(CNN, cnn_params, ensemble_size, CNN, selector_cnn_params)
model.double()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_criterion = torch.nn.BCELoss(reduction='none')
test_criterion = torch.nn.BCELoss()

save_acc(results_path, 'Epoch', ['Train loss', 'Validation loss', 'Validation accuracy', 'Mixed loss', 'Mixed accuracy', 'ACIN loss', 'ACIN accuracy', 'Horodecki loss', 'Horodecki accuracy', 'Bennet loss', 'Bennet accuracy', 'Bisep loss', 'Bisep accuracy'], write_mode='w')

for epoch in range(epoch_num):
    train_loss = train_ensemble(model, device, train_loader, optimizer, train_criterion, epoch, batch_interval)
    val_loss, val_acc = test(model, device, val_loader, test_criterion, "Validation data set", bipart=True)
    mixed_loss, mixed_acc = test(model, device, test_mixed_loader, test_criterion, "Mixed data set", bipart=True)
    acin_loss, acin_acc = test(model, device, test_acin_loader, test_criterion, "ACIN data set", bipart=True) 
    horodecki_loss, horodecki_acc = test(model, device, test_horodecki_loader, test_criterion, "Horodecki data set", bipart=True)
    bennet_loss, bennet_acc = test(model, device, test_bennet_loader, test_criterion, "Bennet data set", bipart=True)   
    biseparable_loss, biseparable_acc = test(model, device, test_biseparable_loader, test_criterion, "Biseparable data set", bipart=True)
    save_acc(results_path, epoch, [train_loss, val_loss, val_acc, mixed_loss, mixed_acc, acin_loss, acin_acc, horodecki_loss, horodecki_acc,\
                            bennet_loss, bennet_acc, biseparable_loss, biseparable_acc])
    torch.save(model.state_dict(), model_path)
