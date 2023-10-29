import sys
sys.path.append('./')

from commons.data.filters import separator_filter
from commons.data.datasets import BipartitionMatricesDataset, DataFilteredSubset
from commons.models.cnns import CNN
from commons.models.separator_classifiers import FancySeparatorEnsembleClassifier
from commons.models.separator_classifiers import FancyClassifier
from commons.models.siamese_networks import VectorSiamese
from commons.models.separators import FancySeparator
from commons.test_utils.base import test
from commons.test_utils.siamese import test_vector_siamese
from commons.train_utils.base import train
from commons.train_utils.siamese import train_vector_siamese
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

val_2xd_dictionary_path = './datasets/3qbits/val_2xd/negativity_bipartitions.txt'
val_2xd_root_dir = './datasets/3qbits/val_2xd/matrices/'

mixed_dictionary_path = './datasets/3qbits/mixed_test/negativity_bipartitions.txt'
mixed_root_dir = './datasets/3qbits/mixed_test/matrices/'

acin_dictionary_path = './datasets/3qbits/acin_test/negativity_bipartitions.txt'
acin_root_dir = './datasets/3qbits/acin_test/matrices/'

horodecki_dictionary_path = './datasets/3qbits/horodecki_test/negativity_bipartitions.txt'
horodecki_root_dir = './datasets/3qbits/horodecki_test/matrices/'

bennet_dictionary_path = './datasets/3qbits/bennet_test/negativity_bipartitions.txt'
bennet_root_dir = './datasets/3qbits/bennet_test/matrices/'

results_dir = './results/3qbits/multi_class_siam_20/nopptes_bisep/'
model_dir = './models/3qbits/multi_class_siam_20/nopptes_bisep/'
model_name = 'weights05_ep10_class_best_val_loss_{}'

# thresholds = [0., 5.e-4, 1.e-3, 2.e-3, 5.e-3, 1.e-2, 2.e-2, 5.e-2]

# thresholds = [0., 1.e-3, 1.e-1]
thresholds = np.geomspace(0.001, 0.1, 20)
thresholds = np.insert(thresholds, 0, 0.)

batch_size = 128
batch_interval = 800
qbits_num = 3
sep_ch = 24
sep_fc_num = 4
epoch_num = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = BipartitionMatricesDataset(train_dictionary_path, train_root_dir, 0.0001)
val_dataset = BipartitionMatricesDataset(val_dictionary_path, val_root_dir, 0.0001)
val_2xd = BipartitionMatricesDataset(val_2xd_dictionary_path, val_2xd_root_dir, 0.0001)
test_mixed_dataset = BipartitionMatricesDataset(mixed_dictionary_path, mixed_root_dir, 0.0001)
test_acin_dataset = BipartitionMatricesDataset(acin_dictionary_path, acin_root_dir, 0.0001)
test_horodecki_dataset = BipartitionMatricesDataset(horodecki_dictionary_path, horodecki_root_dir, 0.0001)
test_bennet_dataset = BipartitionMatricesDataset(bennet_dictionary_path, bennet_root_dir, 0.0001)


separator = FancySeparator(qbits_num, sep_ch, fc_layers = sep_fc_num)
separator.double()
separator.load_state_dict(torch.load(separator_path))
separator.eval()

for i in range(len(thresholds) - 1):
    eval_flags = {
        'val': True,
        'val_2xd': True,
        'mixed': True,
        'acin': True,
        'horodecki': True,
        'bennet': True,
    }
    threshold_range = (thresholds[i], thresholds[i+1])
    print('Threshold range: ', threshold_range)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model_path = model_dir + model_name.format(i) + '.pt'
    results_path = results_dir + model_name.format(i) + '.txt'

    # save_acc(results_path, 'Epoch', ['Train loss', 'Validation loss', 'Validation accuracy', 'Validation 2xd loss', 'Validation 2xd accuracy', 'Mixed loss', 'Mixed accuracy', 'ACIN loss', 'ACIN accuracy', 'Horodecki loss', 'Horodecki accuracy',  'Bennet loss', 'Bennet accuracy', f'Threshold range: {str(threshold_range)}'], write_mode='w')
    save_acc(results_path, 'Epoch', ['Train loss', 'Permutation loss', 'LO loss', 'Validation loss', 'Validation accuracy', 'Validation 2xd loss', 'Validation 2xd accuracy', 'Mixed loss', 'Mixed accuracy', 'ACIN loss', 'ACIN accuracy', 'Horodecki loss', 'Horodecki accuracy',  'Bennet loss', 'Bennet accuracy', f'Threshold range: {str(threshold_range)}'], write_mode='w')

    try:
        train_subset = DataFilteredSubset(train_dataset, lambda x: separator_filter(x, separator, threshold_range))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    except:
        save_acc(results_path, 'No train samples in this range', [], write_mode='a')
        continue

    try:
        val_subset = DataFilteredSubset(val_dataset, lambda x: separator_filter(x, separator, threshold_range))
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    except:
        eval_flags['val'] = False

    try:
        val_2xd_subset = DataFilteredSubset(val_2xd, lambda x: separator_filter(x, separator, threshold_range))
        val_2xd_loader = DataLoader(val_2xd_subset, batch_size=batch_size, shuffle=True)
    except:
        eval_flags['val_2xd'] = False

    try:
        test_mixed_subset = DataFilteredSubset(test_mixed_dataset, lambda x: separator_filter(x, separator, threshold_range))
        test_mixed_loader = DataLoader(test_mixed_subset, batch_size=batch_size, shuffle=True)
    except:
        eval_flags['mixed'] = False

    try:
        test_acin_subset = DataFilteredSubset(test_acin_dataset, lambda x: separator_filter(x, separator, threshold_range))
        test_acin_loader = DataLoader(test_acin_subset, batch_size=batch_size, shuffle=True)
    except:
        eval_flags['acin'] = False

    try:
        test_horodecki_subset = DataFilteredSubset(test_horodecki_dataset, lambda x: separator_filter(x, separator, threshold_range))
        test_horodecki_loader = DataLoader(test_horodecki_subset, batch_size=batch_size, shuffle=True)
    except:
        eval_flags['horodecki'] = False

    try:
        test_bennet_subset = DataFilteredSubset(test_bennet_dataset, lambda x: separator_filter(x, separator, threshold_range))
        test_bennet_loader = DataLoader(test_bennet_subset, batch_size=batch_size, shuffle=True)
    except:
        eval_flags['bennet'] = False

    # model = FancySeparatorEnsembleClassifier(qbits_num, sep_ch, sep_fc_num, train_dataset.bipart_num, 3)
    # model = FancyClassifier(qbits_num, sep_ch, sep_fc_num, 5, train_dataset.bipart_num, 128)
    # model = CNN(qbits_num, train_dataset.bipart_num, 3, 5, 2, 16, ratio_type='sqrt', mode='classifier')
    model = VectorSiamese(qbits_num, train_dataset.bipart_num, 3, 5, 2, 16, ratio_type='sqrt', mode='classifier', biparts_mode='all')
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    val_loss, val_acc = 0., 0.
    val_2xd_loss, val_2xd_acc = 0., 0.
    mixed_loss, mixed_acc = None, None
    acin_loss, acin_acc = None, None
    horodecki_loss, horodecki_acc = None, None
    bennet_loss, bennet_acc = None, None
    
    best_loss = 1e10

    for epoch in range(epoch_num):
        # train_loss = train(model, device, train_loader, optimizer, criterion, epoch, batch_interval)
        train_loss, perm_loss, loc_loss = train_vector_siamese(model, device, train_loader, optimizer, criterion, epoch, batch_interval, loc_op_flag=True, reduced_perms_num=1)
        if eval_flags['val']:
            # val_loss, val_acc = test(model, device, val_loader, criterion, "Validation data set", bipart=True)
            val_loss, val_acc = test_vector_siamese(model, device, val_loader, criterion, "Validation data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=False)
        if eval_flags['val_2xd']:
            # val_loss, val_acc = test(model, device, val_loader, criterion, "Validation data set", bipart=True)
            val_2xd_loss, val_2xd_acc = test_vector_siamese(model, device, val_2xd_loader, criterion, "Validation 2xd data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=False)
        if eval_flags['mixed']:
            # mixed_loss, mixed_acc = test(model, device, test_mixed_loader, criterion, "Mixed data set", bipart=True)
            mixed_loss, mixed_acc = test_vector_siamese(model, device, test_mixed_loader, criterion, "Mixed data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=False)
        if eval_flags['acin']:
            # acin_loss, acin_acc = test(model, device, test_acin_loader, criterion, "ACIN data set", bipart=True)    
            acin_loss, acin_acc = test_vector_siamese(model, device, test_acin_loader, criterion, "ACIN data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=False)
        if eval_flags['horodecki']:
            # horodecki_loss, horodecki_acc = test(model, device, test_horodecki_loader, criterion, "Horodecki data set", bipart=True)
            horodecki_loss, horodecki_acc = test_vector_siamese(model, device, test_horodecki_loader, criterion, "Horodecki data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=False)
        if eval_flags['bennet']:
            # bennet_loss, bennet_acc = test(model, device, test_bennet_loader, criterion, "Bennet data set", bipart=True)
            bennet_loss, bennet_acc = test_vector_siamese(model, device, test_bennet_loader, criterion, "Bennet data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=False)
        
        total_val_loss = val_loss + val_2xd_loss

        if total_val_loss < best_loss:
            best_loss = total_val_loss
            torch.save(model.state_dict(), model_path)
        # save_acc(results_path, epoch, [train_loss, val_loss, val_acc, val_2xd_loss, val_2xd_acc, mixed_loss, mixed_acc, acin_loss, acin_acc, horodecki_loss, horodecki_acc, bennet_loss, bennet_acc, f'Threshold range: {str(threshold_range)}'], write_mode='a')
        save_acc(results_path, epoch, [train_loss, perm_loss, loc_loss, val_loss, val_acc, val_2xd_loss, val_2xd_acc, mixed_loss, mixed_acc, acin_loss, acin_acc, horodecki_loss, horodecki_acc, bennet_loss, bennet_acc, f'Threshold range: {str(threshold_range)}'], write_mode='a')
