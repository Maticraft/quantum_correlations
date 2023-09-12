import os

import numpy as np
import scipy
import torch
from torch.utils.data import Dataset
from qiskit.quantum_info import DensityMatrix

from commons.data.savers import DICTIONARY_NAME, MATRICES_DIR_NAME


class DensityMatricesDataset(Dataset):

    def __init__(self, dictionary, root_dir, metrics, threshold, data_limit = None, format = "npy"):
        self.dictionary = load_dict(dictionary)
        self.root_dir = root_dir
        self.metrics = metrics
        self.threshold = threshold
        self.data_limit = data_limit
        self.format = format

        if self.metrics == "global_entanglement":
            self.label_pos = 3

        elif self.metrics == "von_Neumann":
            self.label_pos = 4

        elif self.metrics == "concurrence":
            self.label_pos = 5

        elif self.metrics == "negativity":
            self.label_pos = 6

        elif self.metrics == "realignment":
            self.label_pos = 7

        elif self.metrics == "numerical_separability":
            self.label_pos = 8

        elif self.metrics == "num_near_zero_eigvals":
            self.label_pos = 9

        elif self.metrics == "discord":
            self.label_pos = 10

        elif self.metrics == "trace":
            self.label_pos = 11

        else:
            raise ValueError('Wrong metrics')

    def __len__(self):
        if self.data_limit != None:
            return self.data_limit
        else:
            return len(self.dictionary)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        matrix_name = os.path.join(self.root_dir, f"{self.dictionary[idx][0]}.{self.format}")
        if self.format == "npy":
            matrix = np.load(matrix_name)
        elif self.format == 'mat':
            matrix_name = matrix_name.with_suffix('.mat')
            matrix = scipy.io.loadmat(matrix_name)['rho']
        else:
            raise ValueError('Wrong format')
        matrix_r = np.real(matrix)
        matrix_im = np.imag(matrix)

        tensor = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0))

        label = float(self.dictionary[idx][self.label_pos])
        if label > self.threshold:
            label = 1
        else:
            label = 0
        label = torch.tensor(label).double()
        label = label.unsqueeze(0)

        return (tensor, label)


class BipartitionMatricesDataset(Dataset):

    def __init__(self, dictionary, root_dir, threshold, data_limit = None, format = "npy"):
        self.dictionary = load_dict(dictionary)[:data_limit]
        self.root_dir = root_dir
        self.data_limit = data_limit
        self.bipart_num = len(self.dictionary[0]) - 1
        self.threshold = threshold
        self.format = format


    def __len__(self):
        if self.data_limit != None:
            return self.data_limit
        else:
            return len(self.dictionary)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = f"{self.dictionary[idx][0]}.{self.format}"
        matrix_name = os.path.join(self.root_dir, filename)
        if self.format == "npy":
            matrix = np.load(matrix_name)
        elif self.format == 'mat':
            matrix_name = matrix_name.with_suffix('.mat')
            matrix = scipy.io.loadmat(matrix_name)['rho']
        else:
            raise ValueError('Wrong format')
        
        matrix_r = np.real(matrix)
        matrix_im = np.imag(matrix)

        tensor = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0))

        label = torch.tensor([1. if float(self.dictionary[idx][i]) > self.threshold else 0. for i in range(1, len(self.dictionary[0]))]).double()

        return (tensor, label)


class DensityMatrixLoader:
    def __init__(self, path, label_idx = 6, threshold = 0.0001, format = "npy"):
        self.path = path
        self.dictionary = load_dict(os.path.join(path, DICTIONARY_NAME))
        self.matrices_dir = os.path.join(path, MATRICES_DIR_NAME)
        self.label_pos = label_idx
        self.threshold = threshold
        self.format = format

    def __getitem__(self, idx):
        filename = f"{self.dictionary[idx][0]}.{self.format}"
        matrix_name = os.path.join(self.matrices_dir, filename)
        if self.format == "npy":
            matrix = np.load(matrix_name)
        elif self.format == 'mat':
            matrix_name = matrix_name.with_suffix('.mat')
            matrix = scipy.io.loadmat(matrix_name)['rho']
        else:
            raise ValueError('Wrong format')

        label = float(self.dictionary[idx][self.label_pos])
        if label > self.threshold:
            label = 1
        else:
            label = 0

        return DensityMatrix(matrix), label
    
    def __len__(self):
        return len(self.dictionary)

    def __iter__(self):
        current_index = 0
        while current_index < len(self.dictionary):
            filename = f"{self.dictionary[current_index][0]}.{self.format}"
            matrix_name = os.path.join(self.matrices_dir, filename)
            if self.format == "npy":
                matrix = np.load(matrix_name)
            elif self.format == 'mat':
                matrix_name = matrix_name.with_suffix('.mat')
                matrix = scipy.io.loadmat(matrix_name)['rho']
            else:
                raise ValueError('Wrong format')
            yield DensityMatrix(matrix)
            current_index += 1


def load_dict(filepath):
    with open(filepath, 'r') as dictionary:
        data = dictionary.readlines()
    parsed_data = [row.rstrip("\n").split(', ') for row in data]
    return parsed_data