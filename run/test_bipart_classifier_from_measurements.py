import sys
sys.path.append('./')

from commons.data.datasets import BipartitionMeasurementDataset
from commons.models.cnns import CNN
from commons.models.siamese_networks import VectorSiamese
from commons.models.measurement import Classifier
from commons.test_utils.base import test
from commons.train_utils.base import train

import os

import torch
from torch.utils.data import DataLoader

from commons.pytorch_utils import save_acc

verified_dataset = True

train_dictionary_path = './datasets/3qbits/train_bisep_no_pptes/negativity_bipartitions.txt'
train_root_dir = './datasets/3qbits/train_bisep_no_pptes/matrices/'

val_dictionary_path = './datasets/3qbits/val_bisep_no_pptes/negativity_bipartitions.txt'
val_root_dir = './datasets/3qbits/val_bisep_no_pptes/matrices/'

pure_dictionary_path = './datasets/3qbits/pure_test/negativity_bipartitions.txt'
pure_root_dir = './datasets/3qbits/pure_test/matrices/'

mixed_dictionary_path = './datasets/3qbits/mixed_test/negativity_bipartitions.txt'
mixed_root_dir = './datasets/3qbits/mixed_test/matrices/'

acin_dictionary_path = './datasets/3qbits/acin_test/negativity_bipartitions.txt'
acin_root_dir = './datasets/3qbits/acin_test/matrices/'

horodecki_dictionary_path = f'./datasets/3qbits/horodecki_test/negativity_bipartitions.txt'
horodecki_root_dir = f'./datasets/3qbits/horodecki_test/matrices/'

bennet_dictionary_path = f'./datasets/3qbits/bennet_test/negativity_bipartitions.txt'
bennet_root_dir = f'./datasets/3qbits/bennet_test/matrices/'

biseparable_dictionary_path = './datasets/3qbits/biseparable_test/negativity_bipartitions.txt'
biseparable_root_dir = './datasets/3qbits/biseparable_test/matrices/'

ghz_dictionary_path = './datasets/3qbits/ghz_test/negativity_bipartitions.txt'
ghz_root_dir = './datasets/3qbits/ghz_test/matrices/'

w_dictionary_path = './datasets/3qbits/w_test/negativity_bipartitions.txt'
w_root_dir = './datasets/3qbits/w_test/matrices/'

results_dir = './results/3qbits/nopptes_bisep_test/'
model_dir = './paper_models/3qbits/nopptes_bisep/'

model_name = 'measurement_class_best_val_paper'
results_file = 'measurement_class_best_val_paper_test.txt'

batch_size = 128
batch_interval = 800
qbits_num = 3
sep_ch = 16
sep_fc_num = 4
epoch_num = 20
input_dim = 4 ** qbits_num
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = BipartitionMeasurementDataset(train_dictionary_path, train_root_dir, 0.0001, data_limit=560000)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = BipartitionMeasurementDataset(val_dictionary_path, val_root_dir, 0.0001)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_pure_dataset = BipartitionMeasurementDataset(mixed_dictionary_path, mixed_root_dir, 0.0001)
test_pure_loader = DataLoader(test_pure_dataset, batch_size=batch_size, shuffle=True)

test_mixed_dataset = BipartitionMeasurementDataset(mixed_dictionary_path, mixed_root_dir, 0.0001)
test_mixed_loader = DataLoader(test_mixed_dataset, batch_size=batch_size, shuffle=True)

test_acin_dataset = BipartitionMeasurementDataset(acin_dictionary_path, acin_root_dir, 0.0001)
test_acin_loader = DataLoader(test_acin_dataset, batch_size=batch_size, shuffle=True)

test_horodecki_dataset = BipartitionMeasurementDataset(horodecki_dictionary_path, horodecki_root_dir, 0.0001)
test_horodecki_loader = DataLoader(test_horodecki_dataset, batch_size=batch_size, shuffle=True)

test_bennet_dataset = BipartitionMeasurementDataset(bennet_dictionary_path, bennet_root_dir, 0.0001)
test_bennet_loader = DataLoader(test_bennet_dataset, batch_size=batch_size, shuffle=True)

test_biseparable_dataset = BipartitionMeasurementDataset(biseparable_dictionary_path, biseparable_root_dir, 0.0001)
test_biseparable_loader = DataLoader(test_biseparable_dataset, batch_size=batch_size, shuffle=True)

test_ghz_dataset = BipartitionMeasurementDataset(ghz_dictionary_path, ghz_root_dir, 0.0001)
test_ghz_loader = DataLoader(test_ghz_dataset, batch_size=batch_size, shuffle=True)

test_w_dataset = BipartitionMeasurementDataset(w_dictionary_path, w_root_dir, 0.0001)
test_w_loader = DataLoader(test_w_dataset, batch_size=batch_size, shuffle=True)

model_path = model_dir + model_name + '.pt'
results_path = results_dir + results_file

model = Classifier(input_dim, train_dataset.bipart_num, 5, 128)
model.double()
model.load_state_dict(torch.load(model_path))

print('Model loaded')

criterion = torch.nn.BCELoss()

os.makedirs(results_dir, exist_ok=True)

save_acc(results_path, '', ['Train loss', 'Train_acc', 'Validation loss', 'Validation accuracy', 'Pure loss', 'Pure accuracy', 'Mixed loss', 'Mixed accuracy', 'ACIN loss', 'ACIN accuracy', 'Horodecki loss', 'Horodecki accuracy', 'Bennet loss', 'Bennet accuracy', 'Bisep loss', 'Bisep accuracy', 'GHZ loss', 'GHZ accuracy', 'W loss', 'W accuracy'], write_mode='w')


train_loss, train_acc = test(model, device, train_loader, criterion, "Train data set", bipart=True)
val_loss, val_acc = test(model, device, val_loader, criterion, "Validation data set", bipart=True)
pure_loss, pure_acc = test(model, device, test_pure_loader, criterion, "Pure data set", bipart=True)
mixed_loss, mixed_acc = test(model, device, test_mixed_loader, criterion, "Mixed data set", bipart=True)
acin_loss, acin_acc = test(model, device, test_acin_loader, criterion, "ACIN data set", bipart=True)    
horodecki_loss, horodecki_acc = test(model, device, test_horodecki_loader, criterion, "Horodecki data set", bipart=True)
bennet_loss, bennet_acc = test(model, device, test_bennet_loader, criterion, "Bennet data set", bipart=True)  
bisep_loss, bisep_acc = test(model, device, test_biseparable_loader, criterion, "Biseparable data set", bipart=True)
ghz_loss, ghz_acc = test(model, device, test_ghz_loader, criterion, "GHZ data set", bipart=True)
w_loss, w_acc = test(model, device, test_w_loader, criterion, "W data set", bipart=True)

save_acc(results_path, '', [train_loss, train_acc, val_loss, val_acc, pure_loss, pure_acc, mixed_loss, mixed_acc, acin_loss, acin_acc, horodecki_loss, horodecki_acc,\
                    bennet_loss, bennet_acc, bisep_loss, bisep_acc, ghz_loss, ghz_acc, w_loss, w_acc])
