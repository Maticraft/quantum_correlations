import numpy as np
import scipy.linalg as la
from math import log2, floor
from itertools import combinations
from functools import reduce
from toqito.state_props.is_separable import is_separable
# from toqito.state_props.is_ppt import is_ppt
from qiskit.quantum_info import entanglement_of_formation, partial_trace, DensityMatrix, entropy, state_fidelity

from commons.data.circuit_ops import extend_matrix, permute_matrix


# Measure of entanglement of each qubit of the system with the remaining (N −1)-qubits
# Average of all the single qubit linear entropies
# purity = Tr[ro^2]
def global_entanglement(dens_matrix):
    N = int(log2(dens_matrix.dim))
    qubits = list(np.arange(0, N))
    s = 0

    for k in range(N):
        qubits_for_trace = [q for q in qubits if q != k]
        s += partial_trace(dens_matrix, qubits_for_trace).purity().real/N

    return 2*(1 - s)


def realignment_3q(dens_matrix, qargs):
    if qargs == [2] or set(qargs) == {0, 1}:
        r = dens_matrix.data.reshape([2,2,2,2,2,2]).transpose([3,0,4,5,1,2]).reshape([4, 16])
    if qargs == [0] or set(qargs) == {1, 2}:
        r = dens_matrix.data.reshape([2,2,2,2,2,2]).transpose([3,4,0,1,5,2]).reshape([16, 4])
    if qargs == [1] or set(qargs) == {0, 2}:
        r = dens_matrix.data.reshape([2,2,2,2,2,2]).transpose([3,0,5,2,4,1]).reshape([16, 4])

    return r


def realignment(dens_matrix, qargs):
    num_qubits = int(log2(dens_matrix.dim))

    # Inverted qubits indexes correction due to qiskit convention 
    qargs = sorted([num_qubits - q - 1 for q in qargs])

    m = 2 ** len(qargs)
    n = 2 ** (num_qubits - len(qargs))
    matrix = dens_matrix.data

    n_qubits = [q for q in list(np.arange(0, num_qubits)) if q not in qargs]

    realigned_matrix = np.zeros((n*n, m*m), dtype=np.complex128)
    state = np.zeros(num_qubits, dtype=int)

    for i in range(n):
        x = state.copy()
        bit_i = bin(i)[2:].zfill(len(n_qubits))
        x[n_qubits] = [int(b) for b in bit_i]

        for j in range(n):
            y = state.copy()
            bit_j = bin(j)[2:].zfill(len(n_qubits))
            y[n_qubits] = [int(b) for b in bit_j]

            sub_matrix = np.zeros((m, m), dtype = np.complex128)
            sub_matrix_indx = np.zeros((2, m*m), dtype=int)

            for k in range(m):
                bit_k = bin(k)[2:].zfill(len(qargs))
                x[qargs] = [int(b) for b in bit_k]

                for l in range(m):
                    bit_l = bin(l)[2:].zfill(len(qargs))                  
                    y[qargs] = [int(b) for b in bit_l]

                    x_ind = int("".join([str(b) for b in list(x)]), 2)
                    y_ind = int("".join([str(b) for b in list(y)]), 2)

                    sub_matrix[k, l] = matrix[x_ind, y_ind]
                    sub_matrix_indx[0, k*m + l] = x_ind
                    sub_matrix_indx[1, k*m + l] = y_ind

            realigned_matrix[j*n+i, :] = vec(sub_matrix)

    return realigned_matrix  


def vec(block_matrix):
    return block_matrix.T.flatten()


# Function to check if two density matrices commute with each other
def comm(dm_0, dm_1, eps = 1.e-8):
    bool_matrix = np.abs((dm_0 @ dm_1) - (dm_1 @ dm_0)) < (np.zeros((dm_0.shape[0], dm_0.shape[1])) + eps)
    return np.all(bool_matrix)


# Zero discord test - true if 0, false if 1
# Performed for some arbitrary bipartition, specified by qargs
def zero_discord(dens_matrix, qargs):
    num_qubits = int(log2(dens_matrix.dim))

    # Inverted qubits indexes correction
    qargs = [num_qubits - q - 1 for q in qargs]

    m = 2 ** len(qargs)
    n = 2 ** (num_qubits - len(qargs))
    matrix = dens_matrix.data

    n_qubits = [q for q in list(np.arange(0, num_qubits)) if q not in qargs]

    state = np.array([0 for i in range(num_qubits)])
    sub_matrices = []

    for i in range(n):
        x = state.copy()
        bit_i = bin(i)[2:].zfill(len(n_qubits))
        x[n_qubits] = [int(b) for b in bit_i]

        for j in range(n):
            y = state.copy()
            bit_j = bin(j)[2:].zfill(len(n_qubits))
            y[n_qubits] = [int(b) for b in bit_j]

            sub_matrix = np.zeros((m, m), dtype = np.complex128)

            for k in range(m):
                bit_k = bin(k)[2:].zfill(len(qargs))
                x[qargs] = [int(b) for b in bit_k]

                for l in range(m):
                    bit_l = bin(l)[2:].zfill(len(qargs))                  
                    y[qargs] = [int(b) for b in bit_l]

                    x_ind = int("".join([str(b) for b in list(x)]), 2)
                    y_ind = int("".join([str(b) for b in list(y)]), 2)

                    sub_matrix[k, l] = matrix[x_ind, y_ind]

            sub_matrices.append(sub_matrix)
    
    for sm in sub_matrices:
        if not comm(sm, sm.conjugate().transpose()):
            return False

    inds = list(combinations(range(len(sub_matrices)), 2))
    for ind in inds:
        if not comm(sub_matrices[ind[0]], sub_matrices[ind[1]]):
            return False

    return True


def bures_dist_for_trace_rec(dens_matrix):
    num_qubits = int(log2(dens_matrix.dim))
    dens_matrices = [partial_trace(dens_matrix, [qi for qi in range(num_qubits) if qi != q]).data for q in range(num_qubits)]
    trace_reconstruction = DensityMatrix(reduce(np.kron, reversed(dens_matrices)))
    fidelity = state_fidelity(dens_matrix, trace_reconstruction)
    bures_distance = 2 * (1 - np.sqrt(fidelity))
    return bures_distance


# Partial transpose operation
# https://en.wikipedia.org/wiki/Peres%E2%80%93Horodecki_criterion
def partial_transpose(dens_matrix, qargs):
    num_qubits = int(log2(dens_matrix.dim))

    # Inverted qubits indexes correction
    qargs = [num_qubits - q - 1 for q in qargs]

    m = 2 ** len(qargs)
    n = 2 ** (num_qubits - len(qargs))
    matrix = dens_matrix.data

    n_qubits = [q for q in list(np.arange(0, num_qubits)) if q not in qargs]

    part_trans_matrix = np.zeros((dens_matrix.dim, dens_matrix.dim), dtype=np.complex128)
    state = np.array([0 for i in range(num_qubits)])

    for i in range(n):
        x = state.copy()
        bit_i = bin(i)[2:].zfill(len(n_qubits))
        x[n_qubits] = [int(b) for b in bit_i]

        for j in range(n):
            y = state.copy()
            bit_j = bin(j)[2:].zfill(len(n_qubits))
            y[n_qubits] = [int(b) for b in bit_j]

            sub_matrix = np.zeros((m, m), dtype = np.complex128)
            sub_matrix_indx = np.zeros((2, m*m), dtype=int)

            for k in range(m):
                bit_k = bin(k)[2:].zfill(len(qargs))
                x[qargs] = [int(b) for b in bit_k]

                for l in range(m):
                    bit_l = bin(l)[2:].zfill(len(qargs))                  
                    y[qargs] = [int(b) for b in bit_l]

                    x_ind = int("".join([str(b) for b in list(x)]), 2)
                    y_ind = int("".join([str(b) for b in list(y)]), 2)

                    sub_matrix[k, l] = matrix[x_ind, y_ind]
                    sub_matrix_indx[0, k*m + l] = x_ind
                    sub_matrix_indx[1, k*m + l] = y_ind

            transposed = sub_matrix.T
            part_trans_matrix[tuple(sub_matrix_indx)] = np.reshape(transposed, -1)    

    return DensityMatrix(part_trans_matrix)        


## Verified implementation from qutip
def partial_transpose_dense(rho, qargs):
    """
    Based on Jonas' implementation using numpy.
    Very fast for dense problems.
    """
    num_qubits = int(log2(rho.dim))

    # Inverted qubits indexes correction
    qargs = [num_qubits - q - 1 for q in qargs]

    mask = np.zeros(num_qubits, dtype=int)
    mask[qargs] = 1
    mask = mask.tolist()

    nsys = len(mask)
    pt_dims = np.arange(2 * nsys).reshape(2, nsys).T
    pt_idx = np.concatenate([[pt_dims[n, mask[n]] for n in range(nsys)],
                            [pt_dims[n, 1 - mask[n]] for n in range(nsys)]])

    data = rho.data.reshape(np.array([rho.dims(), rho.dims()]).flatten()).transpose(pt_idx).reshape(rho.data.shape)

    return DensityMatrix(data, dims=rho.dims())


def KyFan(dens_matrix, qargs):
    rm = realignment(dens_matrix, qargs)
    norm = np.sum(la.svdvals(rm))
    return np.log(norm)


def KyFan_3q(dens_matrix, qargs):
    rm = realignment_3q(dens_matrix, qargs)
    norm = np.sum(la.svdvals(rm))
    return np.log(norm)


def negativity(dens_matrix, qargs):
    part_trans = partial_transpose(dens_matrix, qargs)
    eigvals = np.minimum(np.real(la.eigvals(part_trans.data)), 0)
    return -2*np.sum(eigvals)


def negativity_dense(dens_matrix, qargs):
    part_trans = partial_transpose_dense(dens_matrix, qargs)
    eigvals = np.minimum(np.real(la.eigvals(part_trans.data)), 0)
    return -2*np.sum(eigvals)


def concurrence(dens_matrix, qargs):
    trace = partial_trace(dens_matrix, qargs)
    return 2*(1 - trace.purity().real)


def _zero_discord_bipart(dens_matrix, qubits, comb):
    disc = zero_discord(dens_matrix, qargs= list(comb))
    disc_prim = zero_discord(dens_matrix, qargs= [q for q in qubits if q not in list(comb)])

    if disc and disc_prim:
        zd = 0.
    elif disc or disc_prim:
        zd = 0.5
    else:
        zd = 1.
    return zd


def _realignment_bipart(dens_matrix, qubits, comb):
    nr1 = KyFan(dens_matrix, qargs = list(comb))
    nr2 = KyFan(dens_matrix, qargs = [q for q in qubits if q not in list(comb)])
    nr = (nr1 + nr2) / 2
    return nr


def _negativity_bipart(dens_matrix, ppt, not_ent_qbits, qubits, comb):
    no_comb = [q for q in qubits if q not in list(comb)]
    s1 = negativity_dense(dens_matrix, qargs = list(comb))
    s2 = negativity_dense(dens_matrix, qargs = no_comb)

    is_entangled = [q for q in list(comb) if q not in not_ent_qbits] and [q for q in no_comb if q not in not_ent_qbits]

    s1 = 1. if ppt and (s1 < 0.001) and is_entangled else s1
    s2 = 1. if ppt and (s2 < 0.001) and is_entangled else s2

    neg = (s1 + s2) / 2
    return neg


def _concurrence_bipart(dens_matrix, qubits, comb):
    conc1 = concurrence(dens_matrix, qargs = list(comb))
    conc2 = concurrence(dens_matrix, qargs = [q for q in qubits if q not in list(comb)])
    conc = (conc1 + conc2) / 2
    return conc


def _von_Neumann_entropy_bipart(dens_matrix, qubits, comb):
    trace = partial_trace(dens_matrix, qargs = list(comb))
    trace2 = partial_trace(dens_matrix, qargs = [q for q in qubits if q not in list(comb)])
    ent = (entropy(trace) + entropy(trace2)) / 2
    return ent


# Measure of entanglement as an average of entanglement measures in all possible bi-partitions of the N-qubits system
def global_entanglement_bipartitions(dens_matrix, measure = "von_Neumann", ppt = False, return_separate_outputs = False, not_ent_qbits = []):
    """
    Function to measure the correlations in the system (density matrix).
    Parameters:
        - `dens_matrix`: density matrix representing the system. Accepted type is `qiskit.quantum_info.DensityMatrix`
        - `measure`: one of the following:
            - "von_Neumann" - von Neumann entropy
            - "concurrence" - concurrence
            - "negativity" - 2 * standard negativity
            - "realingment" - reshuffled negativity
            - "discord" - simple measure returning 0 for zero discord and 1 for non-zero discord
            - "trace_reconstruction" - Bures distance between the state and its reconstruction from partial trace
        - `ppt` - global flag to set whether 0 negativity states should be labeled as entangled
        - `ppt_bipart` - bipartition flag analogical to `ppt` flag
        - `not_ent_qbits` - if any of ppt flags is set to True, then one can put indices of not entangled qubits here to not label them as entangled

    Returns:
        if `return_separate_outputs` is set to `False` then only average measure is returned, if `return_separate_outputs` is set to `True` then
        both average and bipart measures are returned 
    """
    measures = {
        "von_Neumann": _von_Neumann_entropy_bipart,
        "concurrence": _concurrence_bipart,
        "negativity": lambda rho, qbits, combination: _negativity_bipart(rho, ppt, not_ent_qbits, qbits, combination),
        "realignment": _realignment_bipart,
        "discord": _zero_discord_bipart,
        "trace_reconstruction": lambda rho, qbits, combination: bures_dist_for_trace_rec(rho),
        "numerical_separability": toqito_bipartite_separability
    }

    N = int(log2(dens_matrix.dim))
    qubits = list(np.arange(0, N)) # e.g. [0, 1, 2] -> 3 qubits
    E = 0
    m = 0

    if return_separate_outputs:
        biparts_out = []

    for k in range(1, floor(N/2) + 1):
        qubits_for_trace = list(combinations(qubits, k)) # e.g. N = 3, k = 1 -> [(0,), (1,), (2,)]

        if k == N/2:
            qubits_for_trace = qubits_for_trace[:int(len(qubits_for_trace)/2)]

        for comb in qubits_for_trace:
            measure_value = measures[measure](dens_matrix, qubits, comb)
            E += measure_value
            if return_separate_outputs:
                biparts_out.append(measure_value)
            m += 1

    if return_separate_outputs:
        return E/m, biparts_out
    else:
        return E/m
    

# Average entanglement of formation in 2 qubits subsystems
def global_entanglement_pairwise(dens_matrix, measure = 'negativity'):
    q_num = int(log2(dens_matrix.dim))
    qubits_for_trace = list(combinations(np.arange(q_num), q_num - 2))
    ent = 0

    for comb in qubits_for_trace:
        trace = partial_trace(dens_matrix, qargs = list(comb))
        
        if measure == 'entanglement_of_formation':
            ent += entanglement_of_formation(trace)
        elif measure == 'negativity':
            ent += negativity_dense(trace, qargs = [0])


    return ent/len(qubits_for_trace)


def bipartitions_num(qubits_num):
    qubits = list(np.arange(qubits_num))
    l = 0
    for k in range(1, floor(qubits_num/2) + 1):
        qubits_for_trace = list(combinations(qubits, k))

        if k == qubits_num/2:
            qubits_for_trace = qubits_for_trace[:int(len(qubits_for_trace)/2)]
                
        l += len(qubits_for_trace)

    return l

def combinations_num(qubits_num):
    qubits = list(np.arange(qubits_num))
    l = 0
    for k in range(1, qubits_num + 1):
        l += len(list(combinations(qubits, k)))
    return l

# manual extension for toqito symmetric measure for separability
def toqito_bipartite_separability(dens_matrix, qubits, comb):
    num_qubits = len(qubits)
    num_qubits_in_comb = len(comb)
    assert num_qubits_in_comb <= num_qubits/2, "Number of qubits in bipartition must be less or equal to half of the number of qubits in the system"
    num_additional_qubits = num_qubits - 2*num_qubits_in_comb
    permutation = [q for q in qubits if q not in list(comb)] + list(comb)
    permuted_matrix = permute_matrix(permutation, dens_matrix)
    if num_additional_qubits > 0:
        permuted_matrix = extend_matrix(permuted_matrix, num_additional_qubits)
    
    rho = permuted_matrix.data
    if is_separable(rho):
        return 0
    return 1
