import sys
sys.path.append('./')

import os

import torch
from torch.utils.data import DataLoader

from commons.data.datasets import BipartitionMatricesDataset
from commons.models.cnns import CNN
from commons.models.siamese_networks import VectorSiamese
from commons.models.separator_classifiers import FancySeparatorEnsembleClassifier
from commons.models.separator_classifiers import FancyClassifier
from commons.test_utils.base import test
from commons.test_utils.siamese import test_vector_siamese
from commons.pytorch_utils import save_acc

siamese_flag = False
verified_dataset = True

batch_size = 128
batch_interval = 800
qbits_num = 3
sep_ch = 16
sep_fc_num = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if verified_dataset:
    results_dir = './results/3qbits/discord/nopptes_bisep/'
    model_dir = './paper_models/3qbits/nopptes_bisep/'
else:
    results_dir = './results/3qbits/discord/negativity_bisep_test/'
    model_dir = './paper_models/3qbits/negativity_bisep/'

if siamese_flag:
    model_name = 'siam_cnn_class_best_val_paper'
    results_file = 'siam_cnn_class_best_val_paper.txt'
else:
    model_name = 'cnn_class_best_val_paper'
    results_file = 'cnn_class_best_val_paper.txt'

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

model_path = model_dir + model_name + '.pt'
results_path = results_dir + results_file

pure_dictionary_path = './datasets/3qbits/pure_test/negativity_bipartitions.txt'
pure_root_dir = './datasets/3qbits/pure_test/matrices/'

mixed_ent_dictionary_path = './datasets/3qbits/mixed_test_bal/negativity_bipartitions.txt'
mixed_disc_dictionary_path = './datasets/3qbits/mixed_test_bal/discord_bipartitions.txt'
mixed_root_dir = './datasets/3qbits/mixed_test_bal/matrices/'

test_pure_dataset = BipartitionMatricesDataset(pure_dictionary_path, pure_root_dir, 0.0001)
test_pure_loader = DataLoader(test_pure_dataset, batch_size=batch_size, shuffle=True)

test_mixed_bal_ent_dataset = BipartitionMatricesDataset(mixed_ent_dictionary_path, mixed_root_dir, 0.0001)
test_mixed_bal_ent_loader = DataLoader(test_mixed_bal_ent_dataset, batch_size=batch_size, shuffle=True)

test_mixed_bal_disc_dataset = BipartitionMatricesDataset(mixed_disc_dictionary_path, mixed_root_dir, 0.0001)
test_mixed_bal_disc_loader = DataLoader(test_mixed_bal_disc_dataset, batch_size=batch_size)


if siamese_flag:
    model = VectorSiamese(qbits_num, test_pure_dataset.bipart_num, 3, 5, 2, 16, ratio_type='sqrt', mode='classifier', biparts_mode='all')
else:
    model = CNN(qbits_num, test_pure_dataset.bipart_num, 3, 5, 2, 16, ratio_type='sqrt', mode='classifier')

model.double()
model.load_state_dict(torch.load(model_path))
print('Model loaded')

criterion = torch.nn.BCELoss()

save_acc(results_path, '', ['Pure loss', 'Pure bal accuracy', 'Pure prob ent', 'Pure prob sep', 'Mixed disc loss', 'Mixed disc bal accuracy', 'Mixed disc prob', 'Mixed zero disc prob', 'Mixed ent loss', 'Mixed ent bal accuracy', 'Mixed ent prob', 'Mixed sep prob'], write_mode='w')

if siamese_flag:
    pure_loss, pure_acc, pure_cm, pure_prob_ent, pure_prob_sep, pure_bal_acc = test_vector_siamese(model, device, test_pure_loader, criterion, "Pure data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=True, confusion_matrix=True, confusion_matrix_dim=2)
    mixed_disc_loss, mixed_disc_acc, mixed_disc_cm, mixed_disc_prob, mixed_zero_disc_prob, mixed_disc_bal_acc = test_vector_siamese(model, device, test_mixed_bal_disc_loader, criterion, "Mixed data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=True, confusion_matrix=True, confusion_matrix_dim=2)
    mixed_ent_loss, mixed_ent_acc, mixed_ent_cm, mixed_ent_prob, mixed_sep_prob, mixed_ent_bal_acc = test_vector_siamese(model, device, test_mixed_bal_ent_loader, criterion, "Mixed data set", bipart='separate', negativity_ext=False, low_thresh=0.5, high_thresh=0.5, decision_point=0.5, balanced_acc=True, confusion_matrix=True, confusion_matrix_dim=2)
  
else:
    pure_loss, pure_acc, pure_cm, pure_prob_ent, pure_prob_sep, pure_bal_acc = test(model, device, test_pure_loader, criterion, "Pure data set", bipart=True, confusion_matrix=True, confusion_matrix_dim=2, balanced_acc=True)
    mixed_disc_loss, mixed_disc_acc, mixed_disc_cm, mixed_disc_prob, mixed_zero_disc_prob, mixed_disc_bal_acc = test(model, device, test_mixed_bal_disc_loader, criterion, "Mixed data set", bipart=True, confusion_matrix=True, confusion_matrix_dim=2, balanced_acc=True)
    mixed_ent_loss, mixed_ent_acc, mixed_ent_cm, mixed_ent_prob, mixed_sep_prob, mixed_ent_bal_acc = test(model, device, test_mixed_bal_ent_loader, criterion, "Mixed data set", bipart=True, confusion_matrix=True, confusion_matrix_dim=2, balanced_acc=True)

save_acc(results_path, '', [pure_loss, pure_bal_acc, pure_prob_ent, pure_prob_sep, mixed_disc_loss, mixed_disc_bal_acc, mixed_disc_prob, mixed_zero_disc_prob, mixed_ent_loss, mixed_ent_bal_acc, mixed_ent_prob, mixed_sep_prob])