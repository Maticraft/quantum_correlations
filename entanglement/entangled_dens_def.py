
import os
import numpy as np

# for negativity label_pos = 6
def entangled_dens(dictionary, threshold, label_pos = 6):
    labels = [1 if float(dictionary[idx][label_pos]) > threshold else 0 for idx in range(len(dictionary))]
    num_entangled = np.sum(np.array(labels))
    return num_entangled / len(dictionary)


def load_dict(filepath):  
    with open(filepath, 'r') as dictionary:
        data = dictionary.readlines()
    parsed_data = [row.rstrip("\n").split(', ') for row in data]

    return parsed_data


filepath = './data/{}qbits/mixed_def_n/fully_ent_all/{}/dictionary.txt'

# Parameters to be modified:
results_path = './results/article_fullent_dens_3q_tot.txt'
qbits = 3
thresh = 0.001
num_states = np.arange(1, 100)


res = []
for n in num_states:
    fp = filepath.format(qbits, n)
    dictionary = load_dict(fp)
    dens_n = entangled_dens(dictionary, thresh)
    res.append([n, dens_n])

np.savetxt(results_path, np.array(res))