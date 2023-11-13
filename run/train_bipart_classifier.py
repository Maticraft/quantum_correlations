import sys
sys.path.append('./')

from commons.data.datasets import BipartitionMatricesDataset
from commons.models.cnns import CNN
from commons.models.siamese_networks import VectorSiamese
from commons.models.separator_classifiers import FancySeparatorEnsembleClassifier
from commons.models.separator_classifiers import FancyClassifier
from commons.models.eigvals_classifier import EigvalsClassifier
from commons.test_utils.base import test
from commons.test_utils.siamese import test_vector_siamese
from commons.train_utils.base import train
from commons.train_utils.siamese import train_vector_siamese

import os

import torch
from torch.utils.data import DataLoader

from commons.pytorch_utils import save_acc

siamese_flag = False
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
    results_dir = './results/3qbits/nopptes_bisep_test/'
    model_dir = './models/3qbits/nopptes_bisep/'
else:
    results_dir = './results/3qbits/negativity_bisep_test/'
    model_dir = './models/3qbits/negativity_bisep/'

if siamese_flag:
    model_name = 'siam_cnn_class_best_val_paper'
    results_file = 'siam_cnn_class_best_val_paper.txt'
else:
    model_name = 'cnn_class_best_val_paper'
    results_file = 'cnn_class_best_val_paper.txt'

batch_size = 128
batch_interval = 800
qbits_num = 3
sep_ch = 16
sep_fc_num = 4
epoch_num = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

siamese_flag = False

train_dataset = BipartitionMatricesDataset(train_dictionary_path, train_root_dir, 0.0001, format='npy', filename_pos=0, data_limit=560000)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = BipartitionMatricesDataset(val_dictionary_path, val_root_dir, 0.0001)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

val_2xd = BipartitionMatricesDataset(val_2xd_dictionary_path, val_2xd_root_dir, 0.0001)
val_2xd_loader = DataLoader(val_2xd, batch_size=batch_size, shuffle=True)

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
# model = EigvalsClassifier(qbits_num, train_dataset.bipart_num, input_channels=2, fc_num=10, linear_transforms=128, hidden_size=1024)

if siamese_flag:
    model = VectorSiamese(qbits_num, train_dataset.bipart_num, 3, 5, 2, 16, ratio_type='sqrt', mode='classifier', biparts_mode='all')
else:
    model = CNN(qbits_num, train_dataset.bipart_num, 3, 5, 2, 16, ratio_type='sqrt', mode='classifier')

model.double()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

model_path = model_dir + model_name + '.pt'
results_path = results_dir + model_name + '.txt'

if siamese_flag:
    save_acc(results_path, 'Epoch', ['Train loss', 'Permutation loss', 'LO loss', 'Validation loss', 'Validation accuracy', 'Validation 2xd loss', 'Validation 2xd accuracy', 'Mixed loss', 'Mixed accuracy', 'ACIN loss', 'ACIN accuracy', 'Horodecki loss', 'Horodecki accuracy', 'Bennet loss', 'Bennet accuracy', ], write_mode='w')
else:
    save_acc(results_path, 'Epoch', ['Train loss', 'Validation loss', 'Validation accuracy', 'Validation 2xd loss', 'Validation 2xd accuracy', 'Mixed loss', 'Mixed accuracy', 'ACIN loss', 'ACIN accuracy', 'Horodecki loss', 'Horodecki accuracy', 'Bennet loss', 'Bennet accuracy'], write_mode='w')

best_loss = 1e10

for epoch in range(epoch_num):
    if siamese_flag:
        train_loss, perm_loss, loc_loss = train_vector_siamese(model, device, train_loader, optimizer, criterion, epoch, batch_interval, loc_op_flag=True, reduced_perms_num=1)
        val_loss, val_acc = test_vector_siamese(model, device, val_loader, criterion, "Validation data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=False)
        val_2xd_loss, val_2xd_acc = test_vector_siamese(model, device, val_2xd_loader, criterion, "Validation 2xd data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=False)
        mixed_loss, mixed_acc = test_vector_siamese(model, device, test_mixed_loader, criterion, "Mixed data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=False)
        acin_loss, acin_acc = test_vector_siamese(model, device, test_acin_loader, criterion, "ACIN data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=False)
        horodecki_loss, horodecki_acc = test_vector_siamese(model, device, test_horodecki_loader, criterion, "Horodecki data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=False)
        bennet_loss, bennet_acc = test_vector_siamese(model, device, test_bennet_loader, criterion, "Bennet data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=False)
        save_acc(results_path, epoch, [train_loss, perm_loss, loc_loss, val_loss, val_acc, val_2xd_loss, val_2xd_acc, mixed_loss, mixed_acc, acin_loss, acin_acc, horodecki_loss, horodecki_acc, bennet_loss, bennet_acc], write_mode='a')
    else:
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, batch_interval)
        val_loss, val_acc = test(model, device, val_loader, criterion, "Validation data set", bipart=True)
        val_2xd_loss, val_2xd_acc = test(model, device, val_2xd_loader, criterion, "Validation 2xd data set", bipart=True)
        mixed_loss, mixed_acc = test(model, device, test_mixed_loader, criterion, "Mixed data set", bipart=True)
        acin_loss, acin_acc = test(model, device, test_acin_loader, criterion, "ACIN data set", bipart=True)    
        horodecki_loss, horodecki_acc = test(model, device, test_horodecki_loader, criterion, "Horodecki data set", bipart=True)
        bennet_loss, bennet_acc = test(model, device, test_bennet_loader, criterion, "Bennet data set", bipart=True)   
        save_acc(results_path, epoch, [train_loss, val_loss, val_acc, val_2xd_loss, val_2xd_acc, mixed_loss, mixed_acc, acin_loss, acin_acc, horodecki_loss, horodecki_acc,\
                            bennet_loss, bennet_acc])
    
    total_val_loss = val_loss + val_2xd_loss

    if total_val_loss < best_loss:
        best_loss = total_val_loss
        torch.save(model.state_dict(), model_path)
