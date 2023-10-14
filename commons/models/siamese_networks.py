from commons.models.separator_classifiers import CombinedClassifier
from commons.models.cnns import CNN


import numpy as np
import torch


from itertools import combinations, permutations
from math import floor


class Siamese(CNN):
    def __init__(self, qbits_num, output_size, conv_num, fc_num, kernel_size, filters_ratio, dilation = 1, ratio_type = None, pooling = None, dropout = False, mode = 'regression'):
        super(Siamese, self).__init__(qbits_num, output_size, conv_num, fc_num, kernel_size, filters_ratio, dilation, ratio_type, pooling, dropout, mode)

    def forward(self, xs):
        out = [super().forward(x) for x in xs]
        return out
    

class VectorSiamese(CNN):
    def __init__(self, qbits_num, output_size, conv_num, fc_num, kernel_size, filters_ratio, dilation = 1, ratio_type = None, pooling = None, dropout = False, mode = 'regression', biparts_mode = 'all', upsample_layers = 0, input_channels = 2, stride = 1, activation = 'relu', cn_in_block = 1, batch_norm = False):
        super(VectorSiamese, self).__init__(qbits_num, output_size, conv_num, fc_num, kernel_size, filters_ratio, dilation, ratio_type, pooling, dropout, mode, upsample_layers, input_channels=input_channels, stride=stride, activation=activation,cn_in_block= cn_in_block, batch_norm= batch_norm)
        self.perms = list(permutations(range(qbits_num)))

        biparts = []
        for k in range(1, floor(qbits_num/2) + 1):
            combs = list(combinations(range(qbits_num), k))
            if k == qbits_num/2:
                biparts += combs[:int(len(combs)/2)]
            else:
                biparts += combs
        self.biparts = biparts

        self.matching_indices = self.find_matching_perms(self.perms, self.biparts, biparts_mode)

    def forward(self, xs):
        if type(xs) == list:
            out = torch.stack([super(VectorSiamese, self).forward(x) for x in xs])
        else:
            out = super(VectorSiamese, self).forward(xs)
        return out

    def find_matching_perms(self, perms, combs, biparts_mode):
        indices = []
        for n in range(1, len(combs[len(combs) - 1]) + 1):
            for i1 in range(len(combs)):
                comb1 = np.array(combs[i1])
                if len(comb1) == n:
                    for j1 in range(len(perms)):
                        perm1 = np.array(perms[j1])
                        for i2 in range(i1, len(combs)):
                            comb2 = np.array(combs[i2])
                            if len(comb2) == n:
                                for j2 in range(j1 + 1, len(perms)):
                                    perm2 = np.array(perms[j2])

                                    if biparts_mode == 'red' and comb1 == comb2 and set(perm1[comb1]) == set(perm2[comb2]):
                                        indices.append(((i1,j1), (i2,j2)))

                                    if biparts_mode == 'all' and set(perm1[comb1]) == set(perm2[comb2]): # and perm1[comb2] == perm2[comb1]:
                                        indices.append(((i1,j1), (i2,j2)))
        return indices


class VectorSiameseCC(CombinedClassifier):
    def __init__(self, qbits_num, output_size, conv_num, fc_num, kernel_size, filters_ratio, dilation = 1, ratio_type = None, pooling = None, dropout = False, mode = 'regression', biparts_mode = 'all', upsample_layers = 0, stride = 1, activation = 'relu', tensor_layers = False, tensor_map = False, sep_ch = 32):
        super(VectorSiameseCC, self).__init__(qbits_num, output_size, conv_num, fc_num, kernel_size, filters_ratio, dilation, ratio_type, pooling, dropout, mode, upsample_layers, stride=stride, activation=activation, tensor_layers=tensor_layers, tensor_map=tensor_map, sep_ch=sep_ch)
        self.perms = list(permutations(range(qbits_num)))

        biparts = []
        for k in range(1, floor(qbits_num/2) + 1):
            combs = list(combinations(range(qbits_num), k))
            if k == qbits_num/2:
                biparts += combs[:int(len(combs)/2)]
            else:
                biparts += combs
        self.biparts = biparts

        self.matching_indices = self.find_matching_perms(self.perms, self.biparts, biparts_mode)

    def forward(self, xs):

        outs = []
        for x in xs:
            out = super().forward(x)
            outs.append(out)

        return torch.stack(outs, dim=0)

    def find_matching_perms(self, perms, combs, biparts_mode):
        indices = []
        for n in range(1, len(combs[len(combs) - 1]) + 1):
            for i1 in range(len(combs)):
                comb1 = np.array(combs[i1])
                if len(comb1) == n:
                    for j1 in range(len(perms)):
                        perm1 = np.array(perms[j1])
                        for i2 in range(i1, len(combs)):
                            comb2 = np.array(combs[i2])
                            if len(comb2) == n:
                                for j2 in range(j1 + 1, len(perms)):
                                    perm2 = np.array(perms[j2])

                                    if biparts_mode == 'red' and comb1 == comb2 and set(perm1[comb1]) == set(perm2[comb2]):
                                        indices.append(((i1,j1), (i2,j2)))

                                    if biparts_mode == 'all' and set(perm1[comb1]) == set(perm2[comb2]): # and perm1[comb2] == perm2[comb1]:
                                        indices.append(((i1,j1), (i2,j2)))
        return indices
