import sys
sys.path.append('./')

from tqdm import tqdm
import os

import numpy as np
from qiskit.quantum_info import state_fidelity

from commons.data.datasets import DensityMatrixLoader

data_path = './datasets/3qbits/train_non_product_separable'
save_file_name = 'statistics.txt'

matrix_loader = DensityMatrixLoader(data_path)

subsample_size = 1000
subsample1 = np.random.choice(len(matrix_loader), subsample_size, replace=False)
subsample2 = np.random.choice(len(matrix_loader), subsample_size, replace=False)

avg_fidelity = {
    'separable': 0,
    'entangled': 0,
    'mixed': 0
}
min_fidelity = {
    'separable': 1,
    'entangled': 1,
    'mixed': 1
}
avg_bures = {
    'separable': 0,
    'entangled': 0,
    'mixed': 0
}
max_bures = {
    'separable': 0,
    'entangled': 0,
    'mixed': 0
}
avg_l2 = {
    'separable': 0,
    'entangled': 0,
    'mixed': 0
}
max_l2 = {
    'separable': 0,
    'entangled': 0,
    'mixed': 0
}
labels_count = {
    'separable': 0,
    'entangled': 0,
    'mixed': 0
}

for idx1 in tqdm(subsample1, desc='Calculating statistics'):
    for idx2 in subsample2:
        matrix1, label1 = matrix_loader[idx1]
        matrix2, label2 = matrix_loader[idx2]
        fidelity = state_fidelity(matrix1, matrix2)
        bures = 2 * (1 - np.sqrt(fidelity))
        l2 = np.linalg.norm(matrix1.data - matrix2.data)

        if label1 == 0 and label2 == 0:
            label_type = 'separable'
        elif label1 == 1 and label2 == 1:
            label_type = 'entangled'
        else:
            label_type = 'mixed'

        avg_fidelity[label_type] += fidelity
        avg_bures[label_type] += bures
        avg_l2[label_type] += l2
        if fidelity < min_fidelity[label_type]:
            min_fidelity[label_type] = fidelity
        if bures > max_bures[label_type]:
            max_bures[label_type] = bures
        if l2 > max_l2[label_type]:
            max_l2[label_type] = l2

        labels_count[label_type] += 1

avg_fidelity['separable'] /= labels_count['separable'] + 1e-10
avg_fidelity['entangled'] /= labels_count['entangled'] + 1e-10
avg_fidelity['mixed'] /= labels_count['mixed'] + 1e-10

avg_bures['separable'] /= labels_count['separable'] + 1e-10
avg_bures['entangled'] /= labels_count['entangled'] + 1e-10
avg_bures['mixed'] /= labels_count['mixed'] + 1e-10

avg_l2['separable'] /= labels_count['separable'] + 1e-10
avg_l2['entangled'] /= labels_count['entangled'] + 1e-10
avg_l2['mixed'] /= labels_count['mixed'] + 1e-10


save_path = os.path.join(data_path, save_file_name)
with open(save_path, 'w') as f:
    # Save results
    f.write('Average fidelity:\n')
    f.write(f'Separable: {avg_fidelity["separable"]}\n')
    f.write(f'Entangled: {avg_fidelity["entangled"]}\n')
    f.write(f'Mixed: {avg_fidelity["mixed"]}\n')
    f.write('\n')
    f.write('Minimum fidelity:\n')
    f.write(f'Separable: {min_fidelity["separable"]}\n')
    f.write(f'Entangled: {min_fidelity["entangled"]}\n')
    f.write(f'Mixed: {min_fidelity["mixed"]}\n')
    f.write('\n')
    f.write('Average Bures distance:\n')
    f.write(f'Separable: {avg_bures["separable"]}\n')
    f.write(f'Entangled: {avg_bures["entangled"]}\n')
    f.write(f'Mixed: {avg_bures["mixed"]}\n')
    f.write('\n')
    f.write('Maximum Bures distance:\n')
    f.write(f'Separable: {max_bures["separable"]}\n')
    f.write(f'Entangled: {max_bures["entangled"]}\n')
    f.write(f'Mixed: {max_bures["mixed"]}\n')
    f.write('\n')
    f.write('Average L2 distance:\n')
    f.write(f'Separable: {avg_l2["separable"]}\n')
    f.write(f'Entangled: {avg_l2["entangled"]}\n')
    f.write(f'Mixed: {avg_l2["mixed"]}\n')
    f.write('\n')
    f.write('Maximum L2 distance:\n')
    f.write(f'Separable: {max_l2["separable"]}\n')
    f.write(f'Entangled: {max_l2["entangled"]}\n')
    f.write(f'Mixed: {max_l2["mixed"]}\n')
    f.write('\n')
    f.write('Labels count:\n')
    f.write(f'Separable: {labels_count["separable"]}\n')
    f.write(f'Entangled: {labels_count["entangled"]}\n')
    f.write(f'Mixed: {labels_count["mixed"]}\n')
    