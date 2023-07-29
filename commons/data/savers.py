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
CONFIDENCE_THRESHOLD = 0.0001

# Save generated density matrix with corresponding metrics to the given destination, depending on the data type
def save_dens_matrix_with_labels(num_qubits, filename, ro, method, entangled_qbits, save_data_dir = "train", ppt = False, separate_bipart = False, not_ent_qbits = [], zero_neg = 'incl', discord = False, trace_reconstruction = False):
    if ppt and zero_neg == 'none':
        raise NotImplementedError("Labeling potential PPTES not implemented for nonzero states")

    data_dir = _make_dir(Path(f"./{DATASETS_DIR_NAME}/{num_qubits}qbits/{save_data_dir}/"))
    matrices_dir = _make_dir(data_dir / MATRICES_DIR_NAME)

    file_path = matrices_dir / filename
    dictionary_path = data_dir / DICTIONARY_NAME

    extra_info = _get_extra_info(ro, trace_reconstruction, discord)
    neg_glob, neg_bipart, disc_glob, disc_bipart = _get_neg_and_disc(ro, ppt, not_ent_qbits)
    
    bipart_metrics, bipart_dict_paths = _get_bipart_metrics(ro, discord, data_dir, neg_bipart, disc_bipart, separate_bipart)

    if zero_neg == 'incl':
        _save_data(file_path, ro, filename, dictionary_path, method, entangled_qbits, all_bipart_metrics=bipart_metrics, bipart_dict_paths=bipart_dict_paths, extra_info=extra_info)

    elif zero_neg == 'only':
        if neg_glob < CONFIDENCE_THRESHOLD:
            _save_data(file_path, ro, filename, dictionary_path, method, entangled_qbits, all_bipart_metrics=bipart_metrics, bipart_dict_paths=bipart_dict_paths, extra_info=extra_info)

    elif zero_neg == 'none':
        l = combinations_num(len(not_ent_qbits))
        if neg_glob >= CONFIDENCE_THRESHOLD and np.count_nonzero(np.array(neg_bipart) <= CONFIDENCE_THRESHOLD) <= l:
            _save_data(file_path, ro, filename, dictionary_path, method, entangled_qbits, all_bipart_metrics=bipart_metrics, bipart_dict_paths=bipart_dict_paths, extra_info=extra_info)
    
    elif zero_neg == 'zero_discord':
        if (disc_glob <= CONFIDENCE_THRESHOLD) or (neg_glob >= CONFIDENCE_THRESHOLD):
            if np.count_nonzero(np.array(neg_bipart) <= CONFIDENCE_THRESHOLD) == np.count_nonzero(np.array(disc_bipart) <= CONFIDENCE_THRESHOLD):
                _save_data(file_path, ro, filename, dictionary_path, method, entangled_qbits, all_bipart_metrics=bipart_metrics, bipart_dict_paths=bipart_dict_paths, extra_info=extra_info)
    
    elif zero_neg == 'zero_discord_only':
        if (disc_glob <= CONFIDENCE_THRESHOLD) or (separate_bipart and (neg_glob >= CONFIDENCE_THRESHOLD)):
            if np.count_nonzero(np.array(neg_bipart) <= CONFIDENCE_THRESHOLD) == np.count_nonzero(np.array(disc_bipart) <= CONFIDENCE_THRESHOLD):
                _save_data(file_path, ro, filename, dictionary_path, method, entangled_qbits, all_bipart_metrics=bipart_metrics, bipart_dict_paths=bipart_dict_paths, extra_info=extra_info)
    else:
        raise ValueError('Wrong value for zero_neg parameter: {}'.format(zero_neg))


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


def _get_neg_and_disc(ro, ppt, not_ent_qbits):
    neg_glob, neg_bipart = global_entanglement_bipartitions(ro, "negativity", ppt, return_separate_outputs=True, not_ent_qbits=not_ent_qbits)
    disc_glob, disc_bipart = global_entanglement_bipartitions(ro, "discord", return_separate_outputs=True)
    return neg_glob, neg_bipart, disc_glob, disc_bipart


def _get_bipart_metrics(ro, discord, data_dir, neg_bipart, disc_bipart, separate_bipart):
    if not separate_bipart:
        return [], []
    neg_dict_path = data_dir / NEGATIVITY_BIPART_DICT_NAME
    disc_dict_path = data_dir / DISCORD_BIPART_DICT_NAME
    num_sep_dict_pah = data_dir / NUMERICAL_SEPARABILITY_BIPART_DICT_NAME
    numerical_separability_bipart = global_entanglement_bipartitions(ro, "numerical_separability", return_separate_outputs=True)

    bipart_metrics = [neg_bipart, numerical_separability_bipart]
    bipart_dict_paths = [neg_dict_path, num_sep_dict_pah]
        
    if discord:
        bipart_metrics.append(disc_bipart)
        bipart_dict_paths.append(disc_dict_path)
    return bipart_metrics, bipart_dict_paths


def _save_data(file_path, ro, filename, dictionary_path, method, entangled_qbits, all_bipart_metrics=[], bipart_dict_paths=[], extra_info=""):
    np.save(file_path, ro.data)
    for bipart_metrics, bipart_dict_path in zip(all_bipart_metrics, bipart_dict_paths):
        _save_bipart_metrics(bipart_metrics, filename, bipart_dict_path)
    _save_common_metrics(dictionary_path, filename, method, entangled_qbits, ro, extra_info)


def _save_bipart_metrics(bipart_metrics, filename, dict_path):
    bipart_metrics = ", ".join([str(bipart_metric) for bipart_metric in bipart_metrics])
    with open(dict_path, "a") as dic:
        dic.write(filename + ", " + bipart_metrics + "\n")


def _save_common_metrics(dictionary_path, filename, method, entangled_qbits, ro, extra_info=""):
    with open(dictionary_path, "a") as dic:
        dic.write(filename + ", " + method + ", " + str(entangled_qbits) + ", " + str(global_entanglement(ro)) + 
        ", " + str(global_entanglement_bipartitions(ro, "von_Neumann")) + ", " + str(global_entanglement_bipartitions(ro, "concurrence")) +
        ", " + str(global_entanglement_bipartitions(ro, "negativity")) + ", " + str(global_entanglement_bipartitions(ro, 'realignment')) + 
        ", " + str(global_entanglement_bipartitions(ro, 'numerical_separability')) + extra_info + "\n")
