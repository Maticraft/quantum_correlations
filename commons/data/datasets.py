import numpy as np
import torch
from torch.utils.data import Dataset

import os


class DensityMatricesDataset(Dataset):

    def __init__(self, dictionary, root_dir, metrics, threshold, data_limit = None):
        self.dictionary = self.load_dict(dictionary)
        self.root_dir = root_dir
        self.metrics = metrics
        self.threshold = threshold
        self.data_limit = data_limit

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

        elif self.metrics == "discord":
            self.label_pos = 8

        elif self.metrics == "trace":
            self.label_pos = 9

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

        matrix_name = os.path.join(self.root_dir, self.dictionary[idx][0] + ".npy")
        matrix = np.load(matrix_name)
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


    def load_dict(self, filepath):
        with open(filepath, 'r') as dictionary:
            data = dictionary.readlines()

        parsed_data = [row.rstrip("\n").split(', ') for row in data]

        return parsed_data


class BipartitionMatricesDataset(Dataset):

    def __init__(self, dictionary, root_dir, threshold, data_limit = None):
        self.dictionary = self.load_dict(dictionary)[:data_limit]
        self.root_dir = root_dir
        self.data_limit = data_limit
        self.bipart_num = len(self.dictionary[0]) - 1
        self.threshold = threshold


    def __len__(self):
        if self.data_limit != None:
            return self.data_limit
        else:
            return len(self.dictionary)


    def __getitem__(self, idx):
      if torch.is_tensor(idx):
        idx = idx.tolist()

      filename = self.dictionary[idx][0] + ".npy"

      matrix_name = os.path.join(self.root_dir, filename)
      matrix = np.load(matrix_name)
      matrix_r = np.real(matrix)
      matrix_im = np.imag(matrix)

      tensor = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0))

      label = torch.tensor([1. if float(self.dictionary[idx][i]) > self.threshold else 0. for i in range(1, len(self.dictionary[0]))]).double()

      return (tensor, label)


    def load_dict(self, filepath):

      with open(filepath, 'r') as dictionary:
        data = dictionary.readlines()

      parsed_data = [row.rstrip("\n").split(', ') for row in data]

      return parsed_data