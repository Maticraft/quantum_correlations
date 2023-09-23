import sys
sys.path.append('./')

from commons.data.datasets import BipartitionMatricesDataset
from commons.models.cnns import CNN

from commons.models.separator_classifiers import FancySeparatorEnsembleClassifier
from commons.models.separator_classifiers import FancyClassifier
from commons.test_utils.base import test
from commons.train_utils.base import train

import os

import torch
from torch.utils.data import DataLoader

from commons.pytorch_utils import save_acc

train_dictionary_path = './datasets/3qbits/train_bisep_discordant/negativity_bipartitions.txt'
train_root_dir = './datasets/3qbits/train_bisep_discordant/matrices/'

val_dictionary_path = './datasets/3qbits/val_bisep_discordant/negativity_bipartitions.txt'
val_root_dir = './datasets/3qbits/val_bisep_discordant/matrices/'

mixed_dictionary_path = './datasets/3qbits/mixed_test/negativity_bipartitions.txt'
mixed_root_dir = './datasets/3qbits/mixed_test/matrices/'

acin_dictionary_path = './datasets/3qbits/acin_test/negativity_bipartitions.txt'
acin_root_dir = './datasets/3qbits/acin_test/matrices/'

results_dir = './results/3qbits/nopptes_bisep_discordant/'
model_dir = './models/3qbits/nopptes_bisep_discordant/'
model_name = 'cnn_class'

batch_size = 128
batch_interval = 800
qbits_num = 3
sep_ch = 16
sep_fc_num = 4
epoch_num = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = BipartitionMatricesDataset(train_dictionary_path, train_root_dir, 0.0001)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = BipartitionMatricesDataset(val_dictionary_path, val_root_dir, 0.0001)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_mixed_dataset = BipartitionMatricesDataset(mixed_dictionary_path, mixed_root_dir, 0.0001)
test_mixed_loader = DataLoader(test_mixed_dataset, batch_size=batch_size, shuffle=True)

test_acin_dataset = BipartitionMatricesDataset(acin_dictionary_path, acin_root_dir, 0.0001)
test_acin_loader = DataLoader(test_acin_dataset, batch_size=batch_size, shuffle=True)

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

save_acc(results_path, 'Epoch', ['Train loss', 'Validation loss', 'Validation accuracy', 'Mixed loss', 'Mixed accuracy', 'ACIN loss', 'ACIN accuracy'], write_mode='w')

for epoch in range(epoch_num):
    train_loss = train(model, device, train_loader, optimizer, criterion, epoch, batch_interval)
    val_loss, val_acc = test(model, device, val_loader, criterion, "Validation data set", bipart=True)
    mixed_loss, mixed_acc = test(model, device, test_mixed_loader, criterion, "Mixed data set", bipart=True)
    acin_loss, acin_acc = test(model, device, test_acin_loader, criterion, "ACIN data set", bipart=True)    
    save_acc(results_path, epoch, [train_loss, val_loss, val_acc, mixed_loss, mixed_acc, acin_loss, acin_acc])
    torch.save(model.state_dict(), model_path)
