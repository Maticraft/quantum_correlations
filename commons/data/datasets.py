import os
from itertools import product

import numpy as np
import scipy
import torch
from torch.utils.data import Dataset, Subset
from qiskit.quantum_info import DensityMatrix

from commons.data.savers import DICTIONARY_NAME, MATRICES_DIR_NAME
from commons.measurement import Measurement, Kwiat
from commons.metrics import bipartitions_num
from commons.pytorch_utils import extend_states


class DensityMatricesDataset(Dataset):

    def __init__(self, dictionary, root_dir, metrics, threshold, data_limit = None, format = "npy", delimiter = ', ', return_metric_value = False):
        self.dictionary = load_dict(dictionary, delimiter)
        self.root_dir = root_dir
        self.metrics = metrics
        self.threshold = threshold
        self.data_limit = data_limit
        self.format = format
        self.return_metric_value = return_metric_value

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
        matrix = self.read_matrix(matrix_name)
        tensor = self.convert_numpy_matrix_to_tensor(matrix)

        label = self.read_label(idx)

        return (tensor, label)
    
    def read_matrix(self, matrix_name):
        if self.format == "npy":
            matrix = np.load(matrix_name)
        elif self.format == 'mat':
            matrix = scipy.io.loadmat(matrix_name)['rho']
        else:
            raise ValueError('Wrong format')
        return matrix
    
    def convert_numpy_matrix_to_tensor(self, matrix: np.ndarray) -> torch.Tensor:
        matrix_r = np.real(matrix)
        matrix_im = np.imag(matrix)
        tensor = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0))
        return tensor
    
    def read_label(self, idx):
        metric_value = float(self.dictionary[idx][self.label_pos])
        if metric_value > self.threshold:
            label = 1
        else:
            label = 0
        label = torch.tensor(label).double()
        label = label.unsqueeze(0)
        if self.return_metric_value:
            return (label, metric_value)
        return label
    

class MeasurementDataset(DensityMatricesDataset):
    def __init__(self, dictionary, root_dir, metrics, threshold, data_limit = None, format = "npy", delimiter = ', ', return_density_matrix = False):
        super().__init__(dictionary, root_dir, metrics, threshold, data_limit = data_limit, format = format, delimiter = delimiter)
        self.return_density_matrix = return_density_matrix

    def __getitem__(self, idx):
        matrix_name = os.path.join(self.root_dir, f"{self.dictionary[idx][0]}.{self.format}")
        matrix = self.read_matrix(matrix_name)
        rho = self.convert_numpy_matrix_to_tensor(matrix)

        matrix_dim = matrix.shape[0]
        # reshape density matrix from (matrix_dim, matrix_dim) to (2, 2, ..., 2) (2*log2(matrix_dim) times)
        num_qubits = int(np.log2(matrix_dim))
        matrix = matrix.reshape([2]*(2*num_qubits))

        measurements = self._get_all_measurements(matrix)
        tensor = torch.from_numpy(measurements)
        label = self.read_label(idx)

        if not self.return_density_matrix:
            return (tensor, label)
        return (rho, tensor, label)
    
    def _get_all_measurements(self, rho_in):
        num_qubits = len(rho_in.shape)//2
        measurement = Measurement(Kwiat, num_qubits)
        m_all = np.array([measurement.measure(rho_in, basis_indices) for basis_indices in self._get_basis_indices(num_qubits)])
        return m_all
    
    def _get_basis_indices(self, num_qubits):
        # it has to be list of all possible basis indices for given number of qubits
        return list(product([0,1,2,3], repeat=num_qubits))



class BipartitionMatricesDataset(Dataset):

    def __init__(self, dictionary, root_dir, threshold, data_limit = None, format = "npy", filename_pos = 0, delimiter = ', ', return_metric_value = False):
        self.dictionary = load_dict(dictionary, delimiter)[:data_limit]
        self.root_dir = root_dir
        self.data_limit = data_limit
        self.bipart_num = len(self.dictionary[0]) - 1 - filename_pos
        self.threshold = threshold
        self.format = format
        self.filename_pos = filename_pos
        self.return_metric_value = return_metric_value

    def __len__(self):
        if self.data_limit != None:
            return self.data_limit
        else:
            return len(self.dictionary)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.read_filename(idx)
        matrix_name = os.path.join(self.root_dir, filename)
        matrix = self.read_matrix(matrix_name)
        tensor = self.convert_numpy_matrix_to_tensor(matrix)
        label = self.read_label(idx)

        return (tensor, label)
    
    def read_filename(self, idx):
        filename = f"{self.dictionary[idx][self.filename_pos]}.{self.format}"
        if not filename.startswith('dens'):
            filename = 'dens' + filename
        return filename
    
    def read_matrix(self, matrix_name):
        if self.format == "npy":
            matrix = np.load(matrix_name)
        elif self.format == 'mat':
            matrix = scipy.io.loadmat(matrix_name)['rho']
        else:
            raise ValueError('Wrong format')
        return matrix
    
    def convert_numpy_matrix_to_tensor(self, matrix):
        matrix_r = np.real(matrix)
        matrix_im = np.imag(matrix)
        tensor = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0))
        return tensor
    
    def read_label(self, idx):
        metric_value = [float(self.dictionary[idx][i]) for i in range(self.filename_pos + 1, len(self.dictionary[0]))]
        label = [1. if float(self.dictionary[idx][i]) > self.threshold else 0. for i in range(self.filename_pos + 1, len(self.dictionary[0]))]
        
        if self.format == 'mat':
            matric_value = self.revert_labels(matric_value)
            label = self.revert_labels(label)
        label = torch.tensor(label).double()
        metric_value = torch.tensor(metric_value).double()
        if self.return_metric_value:
            return (label, metric_value)
        return label

    def revert_labels(self, labels):
        return [1. if label == 0 else 0. for label in labels]
    

class BipartitionMeasurementDataset(BipartitionMatricesDataset):
    def __init__(self, dictionary, root_dir, threshold, data_limit = None, format = "npy", filename_pos = 0, delimiter = ', ', return_density_matrix = False):
        super().__init__(dictionary, root_dir, threshold, data_limit = data_limit, format = format, filename_pos=filename_pos, delimiter = delimiter)
        self.return_density_matrix = return_density_matrix

    def __getitem__(self, idx):
        filename = self.read_filename(idx)
        matrix_name = os.path.join(self.root_dir, filename)
        matrix = self.read_matrix(matrix_name)
        rho = self.convert_numpy_matrix_to_tensor(matrix)

        matrix_dim = matrix.shape[0]
        # reshape density matrix from (matrix_dim, matrix_dim) to (2, 2, ..., 2) (2*log2(matrix_dim) times)
        num_qubits = int(np.log2(matrix_dim))
        matrix = matrix.reshape([2]*(2*num_qubits))

        measurements = self._get_all_measurements(matrix)
        tensor = torch.from_numpy(measurements)
        label = self.read_label(idx)

        if not self.return_density_matrix:
            return (tensor, label)
        return (rho, tensor, label)
    
    def _get_all_measurements(self, rho_in):
        num_qubits = len(rho_in.shape)//2
        measurement = Measurement(Kwiat, num_qubits)
        m_all = np.array([measurement.measure(rho_in, basis_indices) for basis_indices in self._get_basis_indices(num_qubits)])
        return m_all
    
    def _get_basis_indices(self, num_qubits):
        # it has to be list of all possible basis indices for given number of qubits
        return list(product([0,1,2,3], repeat=num_qubits))
    

class ExtendedBipartDataset(Dataset):
    def __init__(self, dataset, desired_num_qubits):
        self.dataset = dataset
        self.desired_num_qubits = desired_num_qubits
        self.bipart_num = bipartitions_num(desired_num_qubits)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tensor, label = self.dataset[idx]
        tensor, label = extend_states(tensor.unsqueeze(0), label.unsqueeze(0), self.desired_num_qubits)
        return (tensor.squeeze(0), label.squeeze(0))


class DensityMatrixLoader:
    def __init__(self, path, label_idx = 6, threshold = 0.0001, format = "npy", delimiter = ', '):
        self.path = path
        self.dictionary = load_dict(os.path.join(path, DICTIONARY_NAME), delimiter)
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


def load_dict(filepath, delimiter = ', '):
    with open(filepath, 'r') as dictionary:
        data = dictionary.readlines()
    parsed_data = [row.lstrip('\t').rstrip("\n").split(delimiter) for row in data]
    return parsed_data


class DataFilteredSubset(Subset):
    def __init__(self, dataset, filter_func):
        indices = [i for i in range(len(dataset)) if filter_func(dataset[i][0])]
        super().__init__(dataset, indices)
        self.bipart_num = len(dataset.dictionary[0]) - 1


class TargetFilteredSubset(Subset):
    def __init__(self, dataset, filter_func):
        indices = [i for i in range(len(dataset)) if filter_func(dataset[i][1])]
        super().__init__(dataset, indices)
        self.bipart_num = len(dataset.dictionary[0]) - 1


class FilteredSubset(Subset):
    def __init__(self, dataset, filter_func):
        indices = [i for i in range(len(dataset)) if filter_func(dataset[i])]
        super().__init__(dataset, indices)
        self.bipart_num = len(dataset.dictionary[0]) - 1
