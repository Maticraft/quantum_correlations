import sys
sys.path.append('./')

from commons.data.datasets import BipartitionMeasurementDataset
from commons.models.measurement import DensityMatrixReconstructor
from commons.test_utils.measurement import test_matrix_reconstructor
from commons.train_utils.measurement import train_matrix_reconstructor

import os

import torch
from torch.utils.data import DataLoader

from commons.pytorch_utils import save_acc

verified_dataset = True

if verified_dataset:
    train_dictionary_path = './datasets/3qbits/train_bisep_no_pptes/negativity_bipartitions.txt'
    train_root_dir = './datasets/3qbits/train_bisep_no_pptes/matrices/'
else:
    train_dictionary_path = './datasets/3qbits/train_bisep_negativity_labeled/negativity_bipartitions.txt'
    train_root_dir = './datasets/3qbits/train_bisep_negativity_labeled/matrices/'

val_dictionary_path = './datasets/3qbits/val_bisep_no_pptes/negativity_bipartitions.txt'
val_root_dir = './datasets/3qbits/val_bisep_no_pptes/matrices/'

val_2xd_dictionary_path = './datasets/3qbits/val_2xd/negativity_bipartitions.txt'
val_2xd_root_dir = './datasets/3qbits/val_2xd/matrices/'

mixed_dictionary_path = './datasets/3qbits/mixed_test/negativity_bipartitions.txt'
mixed_root_dir = './datasets/3qbits/mixed_test/matrices/'

acin_dictionary_path = './datasets/3qbits/acin_test/negativity_bipartitions.txt'
acin_root_dir = './datasets/3qbits/acin_test/matrices/'

horodecki_dictionary_path = f'./datasets/3qbits/horodecki_test/negativity_bipartitions.txt'
horodecki_root_dir = f'./datasets/3qbits/horodecki_test/matrices/'

bennet_dictionary_path = f'./datasets/3qbits/bennet_test/negativity_bipartitions.txt'
bennet_root_dir = f'./datasets/3qbits/bennet_test/matrices/'

if verified_dataset:
    results_dir = './results/3qbits/nopptes_bisep/'
    model_dir = './paper_models/3qbits/nopptes_bisep/'
else:
    results_dir = './results/3qbits/negativity_bisep/'
    model_dir = './paper_models/3qbits/negativity_bisep/'

model_name = 'density_matrix_reconstructor_best_val_paper'
results_file = 'density_matrix_reconstructor_best_val_paper.txt'

batch_size = 128
batch_interval = 100
qbits_num = 3
sep_ch = 16
sep_fc_num = 4
epoch_num = 20
input_dim = 4**qbits_num
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_dataset = BipartitionMeasurementDataset(train_dictionary_path, train_root_dir, 0.0001, format='npy', filename_pos=0, data_limit=560000, return_density_matrix=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = BipartitionMeasurementDataset(val_dictionary_path, val_root_dir, 0.0001, return_density_matrix=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

val_2xd = BipartitionMeasurementDataset(val_2xd_dictionary_path, val_2xd_root_dir, 0.0001, return_density_matrix=True)
val_2xd_loader = DataLoader(val_2xd, batch_size=batch_size, shuffle=True)

test_mixed_dataset = BipartitionMeasurementDataset(mixed_dictionary_path, mixed_root_dir, 0.0001, return_density_matrix=True)
test_mixed_loader = DataLoader(test_mixed_dataset, batch_size=batch_size, shuffle=True)

test_acin_dataset = BipartitionMeasurementDataset(acin_dictionary_path, acin_root_dir, 0.0001, return_density_matrix=True)
test_acin_loader = DataLoader(test_acin_dataset, batch_size=batch_size, shuffle=True)

test_horodecki_dataset = BipartitionMeasurementDataset(horodecki_dictionary_path, horodecki_root_dir, 0.0001, return_density_matrix=True)
test_horodecki_loader = DataLoader(test_horodecki_dataset, batch_size=batch_size, shuffle=True)

test_bennet_dataset = BipartitionMeasurementDataset(bennet_dictionary_path, bennet_root_dir, 0.0001, return_density_matrix=True)
test_bennet_loader = DataLoader(test_bennet_dataset, batch_size=batch_size, shuffle=True)


model = DensityMatrixReconstructor(input_dim, qbits_num, 5, 128)

model.double()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

model_path = model_dir + model_name + '.pt'
results_path = results_dir + model_name + '.txt'

save_acc(results_path, 'Epoch', ['Train loss', 'Validation loss', 'Validation 2xd loss', 'Mixed loss', 'ACIN loss', 'Horodecki loss', 'Bennet loss'], write_mode='w')

best_loss = 1e10

for epoch in range(epoch_num):
    train_loss = train_matrix_reconstructor(model, device, train_loader, optimizer, epoch, batch_interval, criterion)
    val_loss = test_matrix_reconstructor(model, device, val_loader, criterion, "Validation data set")
    val_2xd_loss = test_matrix_reconstructor(model, device, val_2xd_loader, criterion, "Validation 2xd data set")
    mixed_loss = test_matrix_reconstructor(model, device, test_mixed_loader, criterion, "Mixed data set")
    acin_loss = test_matrix_reconstructor(model, device, test_acin_loader, criterion, "ACIN data set")    
    horodecki_loss = test_matrix_reconstructor(model, device, test_horodecki_loader, criterion, "Horodecki data set")
    bennet_loss = test_matrix_reconstructor(model, device, test_bennet_loader, criterion, "Bennet data set")   
    save_acc(results_path, epoch, [train_loss, val_loss, val_2xd_loss, mixed_loss, acin_loss, horodecki_loss, bennet_loss])
    
    total_val_loss = val_loss + val_2xd_loss

    if total_val_loss < best_loss:
        best_loss = total_val_loss
        torch.save(model.state_dict(), model_path)
