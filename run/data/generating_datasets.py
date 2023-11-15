import sys
sys.path.append('./')

from commons.data.generation_functions import *

# Set number of qubits and encoded parameter
qbits = 3
encoded = True
paper = 'entanglement' # 'entanglement', 'discord' or 'custom'
format = 'npy'

if paper == 'custom':
    # Generate train sets
    _ = generate_train_discordant(qbits, encoded, indx = 0, label_ppt = False, save_data_dir = 'train_bisep_discordant', examples_ratio = 1., zero_neg = 'none', biseparable=True, format=format, discord=True)
    _ = generate_pure_train_balanced(qbits, encoded, indx = 0, save_data_dir = 'train_pure', examples_ratio = 2., max_num_ps = None, format=format)
    _ = generate_train_balanced(qbits, encoded, indx=0, label_ppt=False, save_data_dir='train_no_sep_disc', examples_ratio=1., max_num_ps=None, zero_neg='zero_discord', qubits_glob=9, biseparable=True, format=format)
    # Generate validation set
    _ = generate_train_balanced(qbits, encoded, indx = 0, label_ppt = True, save_data_dir = 'val_bisep_weakly_nnze_neg', examples_ratio = 0.1, max_num_ps = None, zero_neg = 'incl', qubits_glob = 9, biseparable=True, format=format)
    _ = generate_train_discordant(qbits, encoded, indx = 0, label_ppt = False, save_data_dir = 'val_bisep_discordant', examples_ratio = 0.1, zero_neg = 'none', biseparable=True, format=format, discord=True)
    _ = generate_pure_train_balanced(qbits, encoded, indx = 0, save_data_dir = 'val_pure', examples_ratio = 0.2, max_num_ps = None, format=format)
    _ = generate_train_balanced(qbits, encoded, indx=0, label_ppt=False, save_data_dir='val_no_sep_disc', examples_ratio=0.1, max_num_ps=None, zero_neg='zero_discord', qubits_glob=9, biseparable=True, format=format)

    # # Generate test sets
    _ = generate_mixed_test_set(qbits, encoded, indx = 0, save_data_dir = 'mixed_test_sep_filter', format=format, separator_loss_range=[0.001, 0.01])
    _ = generate_mixed_reduced_test(qbits, encoded, indx = 0, save_data_dir = 'mixed_red_test_sep_filter_reduced', format=format, separator_loss_range=[0.001, 0.01])
    _ = generate_mixed_reduced_train_balanced(qbits, encoded, indx = 0, save_data_dir = 'mixed_red_train_sep_filter_reduced', zero_neg='none', format=format, separator_loss_range=[0.001, 0.01])
    _ = generate_max_rank_weak_pptes(qbits, encoded, indx = 0, save_data_dir = 'test_max_rank_weak_pptes', qubits_glob=9, zero_neg='only', label_potent_ppt=False)

if paper == 'discord':
    # Generate full train set
    _ = generate_full_train_separable(qbits, encoded, indx = 0, save_data_dir = 'train_separable', examples_ratio = 1., discord = True, qubits_glob = 9, format=format)
    # Generate non product separable train set
    _ = generate_train_non_product_separable(qbits, encoded, indx = 0, save_data_dir = 'train_non_product_separable', examples_ratio = 1., discord = True, format=format)
    # Generate validation set
    _ = generate_train_separable(qbits, encoded, indx = 0, save_data_dir = 'val_separable', examples_ratio = 0.1, discord = True, qubits_glob = 9, format=format)
    # Generate test sets
    _ = generate_pure_test(qbits, encoded, indx = 0, save_data_dir = 'pure_test', discord = True)
    _ = generate_mixed_balanced_test_set(qbits, encoded, indx = 0, save_data_dir = 'mixed_test_bal', max_num_ps = None, discord = True, permute = False, format=format)

if paper == 'entanglement':
    # Generate train sets
    _ = generate_train_balanced(qbits, encoded, indx = 0, label_ppt = False, save_data_dir = 'train_bisep_negativity_labeled', examples_ratio = 1., max_num_ps = None, zero_neg = 'incl', qubits_glob = 9, biseparable=True, format=format)
    _ = generate_train_balanced(qbits, encoded, indx = 0, label_ppt = False, save_data_dir = 'train_bisep_no_pptes', examples_ratio = 1., max_num_ps = None, zero_neg = 'none', qubits_glob = 9, biseparable=True, format=format)
    _ = generate_max_rank_weak_pptes(qbits, encoded, indx = 1150000, save_data_dir = 'train_bisep_no_pptes', qubits_glob=9, zero_neg='incl') # substitute indx with correct one after generating the train_bisep_no_pptes dataset

    # Generate validation set
    _ = generate_train_balanced(qbits, encoded, indx = 0, label_ppt = False, save_data_dir = 'val_bisep_no_pptes', examples_ratio = 0.1, max_num_ps = None, zero_neg = 'none', qubits_glob = 9, biseparable=True, format=format)
    _ = generate_2xd(qbits, encoded, indx = 0, save_data_dir = 'val_2xd', examples_ratio = 1., discord=True, format=format)

    # Generate test sets
    _ = generate_pure_test(qbits, encoded, indx = 0, save_data_dir = 'pure_test', format=format)
    _ = generate_mixed_test_set(qbits, encoded, indx = 0, save_data_dir = 'mixed_test', format=format)
    _ = generate_acin(qbits, encoded, indx = 0, save_data_dir = 'acin_test', format=format)
    _ = generate_horodecki(qbits, encoded, indx = 0, save_data_dir = 'horodecki_test', format=format)
    _ = generate_bennet(qbits, encoded, indx = 0, save_data_dir = 'bennet_test', format=format)
    _ = generate_ghz(qbits, encoded, indx = 0, save_data_dir = 'ghz_test', format=format)
    _ = generate_w(qbits, encoded, indx = 0, save_data_dir = 'w_test', format=format)
