import sys
sys.path.append('D:/phd/quantum_correlations/quantum_correlations')

from commons.data.generation_functions import *

# Set number of qubits and encoded parameter
qbits = 3
encoded = True
paper = None # 'entanglement' or 'discord'

if paper == 'entanglement':
    # Generate train sets
    _ = generate_train_balanced(qbits, encoded, indx = 0, label_ppt = True, save_data_dir = 'train_weakly_labeled', examples_ratio = 1., max_num_ps = None, zero_neg = 'incl', qubits_glob = 9)
    _ = generate_train_balanced(qbits, encoded, indx = 0, label_ppt = False, save_data_dir = 'train_negativity_labeled', examples_ratio = 1., max_num_ps = None, zero_neg = 'incl', qubits_glob = 9)
    _ = generate_train_balanced(qbits, encoded, indx = 0, label_ppt = False, save_data_dir = 'train_no_pptes', examples_ratio = 1., max_num_ps = None, zero_neg = 'none', qubits_glob = 9)
    # Generate validation set
    _ = generate_train_balanced(qbits, encoded, indx = 0, label_ppt = False, save_data_dir = 'val_no_pptes', examples_ratio = 0.1, max_num_ps = None, zero_neg = 'none', qubits_glob = 9)
    # Generate test sets
    _ = generate_pure_test(qbits, encoded, indx = 0, save_data_dir = 'pure_test')
    _ = generate_mixed_test_set(qbits, encoded, indx = 0, save_data_dir = 'mixed_test')
    _ = generate_acin(qbits, encoded, indx = 0, save_data_dir = 'acin_test')
    _ = generate_horodecki(qbits, encoded, indx = 0, save_data_dir = 'horodecki_test')
    _ = generate_bennet(qbits, encoded, indx = 0, save_data_dir = 'bennet_test')
if paper == 'discord':
    # Generate full train set
    _ = generate_full_train_separable(qbits, encoded, indx = 0, save_data_dir = 'train_separable', examples_ratio = 1., discord = True, qubits_glob = 9)
    # Generate non product separable train set
    _ = generate_train_non_product_separable(qbits, encoded, indx = 0, save_data_dir = 'train_non_product_separable', examples_ratio = 1., discord = True)
    # Generate validation set
    _ = generate_train_separable(qbits, encoded, indx = 0, save_data_dir = 'val_separable', examples_ratio = 0.1, discord = True, qubits_glob = 9)
    # Generate test sets
    _ = generate_pure_test(qbits, encoded, indx = 0, save_data_dir = 'pure_test', discord = True)
    _ = generate_mixed_balanced_test_set(qbits, encoded, indx = 0, save_data_dir = 'mixed_test_bal', max_num_ps = None, discord = True, permute = False)


# _ = generate_acin(qbits, encoded, indx = 0, save_data_dir = 'acin_test', examples_ratio=0.1)
# _ = generate_horodecki(qbits, encoded, indx = 0, save_data_dir = 'horodecki_test', examples_ratio=0.1)
# _ = generate_bennet(qbits, encoded, indx = 0, save_data_dir = 'bennet_test', examples_ratio=0.1)

_ = generate_acin(qbits, encoded, indx = 0, save_data_dir = 'acin_test', examples_ratio=0.1, discord=True)