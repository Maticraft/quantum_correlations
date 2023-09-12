import os
from pathlib import Path

import numpy as np

from commons.metrics import combinations_num, global_entanglement_bipartitions, global_entanglement


DATASETS_DIR_NAME = "datasets"
MATRICES_DIR_NAME = "matrices"
DICTIONARY_NAME = "dictionary.txt"
NEGATIVITY_BIPART_DICT_NAME = "negativity_bipartitions.txt"
DISCORD_BIPART_DICT_NAME = "discord_bipartitions.txt"
NUMERICAL_SEPARABILITY_BIPART_DICT_NAME = "numerical_separability_bipartitions.txt"
NUM_NEAR_ZERO_EIGVALS_BIPART_DICT_NAME = "num_near_zero_eigvals_bipartitions.txt"
CONFIDENCE_THRESHOLD = 0.0001

# Save generated density matrix with corresponding metrics to the given destination, depending on the data type
def save_dens_matrix_with_labels(num_qubits, filename, ro, method, entangled_qbits, save_data_dir = "train", ppt = False, separate_bipart = False, not_ent_qbits = [], zero_neg = 'incl', discord = False, trace_reconstruction = False, num_near_zero_eigvals = None):
    if ppt and zero_neg == 'none':
        raise NotImplementedError("Labeling potential PPTES not implemented for nonzero states")

    data_dir = _make_dir(Path(f"./{DATASETS_DIR_NAME}/{num_qubits}qbits/{save_data_dir}/"))
    matrices_dir = _make_dir(data_dir / MATRICES_DIR_NAME)

    file_path = matrices_dir / filename
    dictionary_path = data_dir / DICTIONARY_NAME

    extra_info = _get_extra_info(ro, trace_reconstruction, discord)
    neg_glob, neg_bipart, disc_glob, disc_bipart = _get_neg_and_disc(ro)
    
    bipart_metrics, bipart_dict_paths = _get_bipart_metrics(ro, discord, data_dir, separate_bipart, ppt, not_ent_qbits)
    
    if num_near_zero_eigvals_matches(num_near_zero_eigvals, bipart_metrics):
        if zero_neg == 'incl':
            _save_data(file_path, ro, filename, dictionary_path, method, entangled_qbits, all_bipart_metrics=bipart_metrics, bipart_dict_paths=bipart_dict_paths, extra_info=extra_info, ppt=ppt)

        elif zero_neg == 'only':
            if neg_glob < CONFIDENCE_THRESHOLD:
                _save_data(file_path, ro, filename, dictionary_path, method, entangled_qbits, all_bipart_metrics=bipart_metrics, bipart_dict_paths=bipart_dict_paths, extra_info=extra_info, ppt=ppt)

        elif zero_neg == 'none':
            comb_num = combinations_num(len(not_ent_qbits))
            if neg_glob >= CONFIDENCE_THRESHOLD and np.count_nonzero(np.array(neg_bipart) <= CONFIDENCE_THRESHOLD) <= comb_num:
                _save_data(file_path, ro, filename, dictionary_path, method, entangled_qbits, all_bipart_metrics=bipart_metrics, bipart_dict_paths=bipart_dict_paths, extra_info=extra_info, ppt=ppt)
        
        elif zero_neg == 'zero_discord':
            if (disc_glob <= CONFIDENCE_THRESHOLD) or (neg_glob >= CONFIDENCE_THRESHOLD):
                if np.count_nonzero(np.array(neg_bipart) <= CONFIDENCE_THRESHOLD) == np.count_nonzero(np.array(disc_bipart) <= CONFIDENCE_THRESHOLD):
                    _save_data(file_path, ro, filename, dictionary_path, method, entangled_qbits, all_bipart_metrics=bipart_metrics, bipart_dict_paths=bipart_dict_paths, extra_info=extra_info, ppt=ppt)
        
        elif zero_neg == 'zero_discord_only':
            if (disc_glob <= CONFIDENCE_THRESHOLD) or (separate_bipart and (neg_glob >= CONFIDENCE_THRESHOLD)):
                if np.count_nonzero(np.array(neg_bipart) <= CONFIDENCE_THRESHOLD) == np.count_nonzero(np.array(disc_bipart) <= CONFIDENCE_THRESHOLD):
                    _save_data(file_path, ro, filename, dictionary_path, method, entangled_qbits, all_bipart_metrics=bipart_metrics, bipart_dict_paths=bipart_dict_paths, extra_info=extra_info, ppt=ppt)
        else:
            raise ValueError('Wrong value for zero_neg parameter: {}'.format(zero_neg))


def num_near_zero_eigvals_matches(num_near_zero_eigvals, bipart_metrics):
    if num_near_zero_eigvals is not None:
        num_near_zero_eigvals_metrics_idx = 2
        num_near_zero_eigvals_metrics = bipart_metrics[num_near_zero_eigvals_metrics_idx]
        return np.all(np.array(num_near_zero_eigvals_metrics) == num_near_zero_eigvals) # check if all elements are equal
    return True


def _make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def _get_extra_info(ro, trace_reconstruction, discord):
    extra_info = ""
    if trace_reconstruction:
        trace_rec_glob = global_entanglement_bipartitions(ro, "trace_reconstruction")
        extra_info = ", " + str(trace_rec_glob) + extra_info
    if discord:
        disc_glob = global_entanglement_bipartitions(ro, "discord")
        extra_info = ", " + str(disc_glob) + extra_info
    return extra_info


def _get_neg_and_disc(ro):
    neg_glob, neg_bipart = global_entanglement_bipartitions(ro, "negativity", return_separate_outputs=True)
    disc_glob, disc_bipart = global_entanglement_bipartitions(ro, "discord", return_separate_outputs=True)
    return neg_glob, neg_bipart, disc_glob, disc_bipart


def _get_bipart_metrics(ro, discord, data_dir, separate_bipart, ppt, not_ent_qbits):
    if not separate_bipart:
        return [], []
    neg_dict_path = data_dir / NEGATIVITY_BIPART_DICT_NAME
    disc_dict_path = data_dir / DISCORD_BIPART_DICT_NAME
    num_sep_dict_path = data_dir / NUMERICAL_SEPARABILITY_BIPART_DICT_NAME
    num_near_zero_eigvals_dict_path = data_dir / NUM_NEAR_ZERO_EIGVALS_BIPART_DICT_NAME
    _, neg_bipart = global_entanglement_bipartitions(ro, "negativity", ppt, return_separate_outputs=True, not_ent_qbits=not_ent_qbits)
    _, numerical_separability_bipart = global_entanglement_bipartitions(ro, "numerical_separability", ppt, return_separate_outputs=True, not_ent_qbits=not_ent_qbits)
    _, num_near_zero_eigvals_bipart = global_entanglement_bipartitions(ro, "near_zero_eigvals", return_separate_outputs=True)

    bipart_metrics = [neg_bipart, numerical_separability_bipart, num_near_zero_eigvals_bipart]
    bipart_dict_paths = [neg_dict_path, num_sep_dict_path, num_near_zero_eigvals_dict_path]
        
    if discord:
        _, disc_bipart = global_entanglement_bipartitions(ro, "discord", ppt, return_separate_outputs=True)
        bipart_metrics.append(disc_bipart)
        bipart_dict_paths.append(disc_dict_path)
    return bipart_metrics, bipart_dict_paths


def _save_data(file_path, ro, filename, dictionary_path, method, entangled_qbits, all_bipart_metrics=[], bipart_dict_paths=[], extra_info="", ppt=False):
    np.save(file_path, ro.data)
    for bipart_metrics, bipart_dict_path in zip(all_bipart_metrics, bipart_dict_paths):
        _save_bipart_metrics(bipart_metrics, filename, bipart_dict_path)
    _save_common_metrics(dictionary_path, filename, method, entangled_qbits, ro, extra_info, ppt)


def _save_bipart_metrics(bipart_metrics, filename, dict_path):
    bipart_metrics = ", ".join([str(bipart_metric) for bipart_metric in bipart_metrics])
    with open(dict_path, "a") as dic:
        dic.write(filename + ", " + bipart_metrics + "\n")


def _save_common_metrics(dictionary_path, filename, method, entangled_qbits, ro, extra_info="", ppt=False):
    with open(dictionary_path, "a") as dic:
        dic.write(filename + ", " + method + ", " + str(entangled_qbits) + ", " + str(global_entanglement(ro)) + 
        ", " + str(global_entanglement_bipartitions(ro, "von_Neumann")) + ", " + str(global_entanglement_bipartitions(ro, "concurrence")) +
        ", " + str(global_entanglement_bipartitions(ro, "negativity", ppt)) + ", " + str(global_entanglement_bipartitions(ro, 'realignment')) + 
        ", " + str(global_entanglement_bipartitions(ro, 'numerical_separability', ppt)) + ", " + str(global_entanglement_bipartitions(ro, 'near_zero_eigvals')) + 
        extra_info + "\n")
