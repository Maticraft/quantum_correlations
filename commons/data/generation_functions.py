from functools import reduce
from operator import mul
import random
from itertools import combinations, permutations

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector

from commons.data.pure_states_generator import PureStatesGenerator
from commons.data.mixed_def_states_generator import MixedDefStatesGenerator
from commons.data.mixed_reduced_states_generator import MixedReducedStatesGenerator, ReducedMethods
from commons.data.pptes_generator import PPTESGenerator
from commons.data.savers import save_dens_matrix_with_labels
from commons.data.circuit_ops import local_randomize_matrix, permute_matrix


# generation of quantum state (density matrix) for specified parameters
def generate_parametrized_qs(qubits_num, a, c, fi2, fi3 = 0, random_amplitude_swap = False):
    """
    Generate density matrix with given params:
    - `qubits_num`: number of qubits in the system
    - `a`: amplitude of the |0>, it can be either single `float` or `list` of `float`s. 
    In the second case, each element of the list corresponds to the different qubit
    - `c`: decoherence parameter ranging from 0 to 1, similarly like `a` it can be either single float `list` of `float`s
    - `fi2`: `list` of phases for basis states with 'two ones' e.g. |11> in case of 2 qubits or |110>, |101> and |011> for 3 qubits system.
    Notice that for |111> basis state combination of all 3 phases is taken into account. If single fi2 is provided, then all phases are set to fi2.  
    - `fi3`: `list` of phases for basis states with 'three ones' e.g. |111> in case of 3 qubits. If single fi3 is provided, then all phases are set to fi3. 
    """
    if type(a) == list:
        assert len(a) == qubits_num, "Wrong dimension of a"
    else:
        a = np.ones(qubits_num)*a
    if type(c) == list:
        assert len(c) == qubits_num, "Wrong dimension of c"
    else:
        c = np.ones(qubits_num)*c
    if type(fi2) == list:
        assert len(fi2) == len(list(combinations(range(qubits_num), 2))), "Wrong dimension of fi"
    else:
        fi2 = np.ones(len(list(combinations(range(qubits_num), 2))))*fi2
    if type(fi3) == list:
        assert len(fi3) == len(list(combinations(range(qubits_num), 3))), "Wrong dimension of fi"
    else:
        fi3 = np.ones(len(list(combinations(range(qubits_num), 3))))*fi3

    b = np.sqrt(1 - np.power(np.absolute(a), 2))

    if random_amplitude_swap:
        num_of_swaps = random.randint(0, qubits_num)
        swap_inds = random.sample(range(qubits_num), num_of_swaps)
        for i in swap_inds:
            a[i], b[i] = b[i], a[i]

    qubits_states_str = [str(bin(i)[2:].zfill(qubits_num)) for i in range(2**qubits_num)] # list of binary strings representing qubits states e.g. ['00', '01', '10', '11']
    qubits_states = [[int(z[x]) for x in range(qubits_num)] for z in qubits_states_str] # list of lists representing qubits states e.g. [[0, 0], [0, 1], [1, 0], [1, 1]]
    sv = [reduce(mul, [list(zip(a, b))[i][j] for i, j in enumerate(qubits_states[x])]) for x in range(2**qubits_num)]

    _entangle_sv_with_phase(sv, fi2, qubits_num, qubits_states, state_value_count=2)
    _entangle_sv_with_phase(sv, fi3, qubits_num, qubits_states, state_value_count=3)

    pure_state = Statevector(sv)
    rho = DensityMatrix(pure_state).data

    for i in range(2**qubits_num):
        for j in range(2**qubits_num):
            for k in range(qubits_num):
                if i % (2**(k+1)) < (2**k) and j % (2**(k+1)) >= (2**k):
                    rho[i, j] *= c[k]
                    rho[j, i] *= np.conjugate(c[k])

    return DensityMatrix(rho)


def _entangle_sv_with_phase(sv, phase, qubits_num, qubits_states, state_value = 1, state_value_count = 2):
    combs = list(combinations(range(qubits_num), state_value_count))
    for i in range(2**qubits_num):
        if len(np.where(np.array(qubits_states[i]) == state_value)[0]) >= 2:
            qubits_combs = list(combinations(np.where(np.array(qubits_states[i]) == state_value)[0], state_value_count))
            for j, comb in enumerate(combs):
                if comb in qubits_combs:
                    sv[i] *= np.exp(-1.j * phase[j])


def generate_parametrized_np_qs(a, p, fi, c, qubits_num = 3):
    """
    Generate density matrix with given params:
    - `qubits_num`: number of qubits in the system
    - `a`: parameter determinig rotation of basis for given qubit, it can be either single `float` or `list` of `float`s. 
    In the second case, each element of the list corresponds to the different qubit
    - `c`: decoherence parameter ranging from 0 to 1, similarly like `a` it can be either single float `list` of `float`s
    - `fi`: `list` of phases for basis state |1> for all qubits. It can be either single `float` or `list` of `float`s.
    - `p`: probability of taking |0><0| density matrix
    """
    a = np.ones(qubits_num)*a
    fi = np.ones(qubits_num)*fi

    # Initialize basis states
    rho0 = np.array([[1, 0], [0, 0]])
    rho1 = np.array([[0, 0], [0, 1]])
     
    # Construct unitary matrices for each qubit
    U0 = [np.array([[a_i, np.sqrt(1 - a_i**2)], [np.conjugate(np.sqrt(1 - a_i**2)), -np.conjugate(a_i)]]) for a_i in a]
    U1 = [np.array([[np.exp(-1.j * fi_i/2), 0], [0, np.exp(1.j * fi_i/2)]]) @ U0_i for fi_i, U0_i in zip(fi, U0)]

    # Construct 'qubits_num'-qubit density matrix
    rho0 = reduce(np.kron, [rho0] * qubits_num)
    rho1 = reduce(np.kron, [rho1] * qubits_num)

    # Construct 'qubits_num'-qubit unitary matrix
    U0 = reduce(np.kron, U0)
    U1 = reduce(np.kron, U1)

    # Construct density matrix
    rho = p * U0 @ rho0 @ U0.conj().T + (1 - p) * U1 @ rho1 @ U1.conj().T

    # Introduce entanglement by increasing the probability of the most probable eigenstate
    # 1) Get the Schmidt decomposition of the density matrix
    # 2) Get the most probable eigenstate
    # 3) Increase the probability of the most probable eigenstate (maintaining the trace norm)
    # 4) Reconstruct the density matrix
    E, V = np.linalg.eigh(rho)
    E = np.diag(E)
    E[-1, -1] += c
    E /= np.trace(E)
    rho = V @ E @ V.conj().T
    return DensityMatrix(rho)


# DISCORD PAPER TRAIN PURE SEPARABLE 2/2 (examples = 30000, pure = True, discord = True, separable = True)
def generate_parametrized(qbits, encoded, indx = 0, save_data_dir = 'parametrized', examples = 20000, local_randomization = True, pure=False, zero_neg = 'incl', discord = False, permute = False, separable = False):
    a_real_range = np.arange(0., 1.01, 0.01)
    if not pure:
        c_real_range = np.arange(0., 1.01, 0.01)
    
    fi_bool = np.array([0., 1.])
    fi_range = np.arange(0, 2*np.pi + 0.01, 0.01)
    fi_len = len(list(combinations(range(qbits), 2)))
    for i in range(examples):
        a = np.random.choice(a_real_range, qbits, replace=True)
        if not pure:
            c = np.random.choice(c_real_range, qbits, replace=True)
            if not separable:
                fi = np.random.choice(fi_range, fi_len, replace=True)
        else:
            c = 1.
            if not separable:
                fi = np.random.choice(fi_bool, fi_len, replace=True)
                fi[fi > 0.] = np.random.choice(fi_range, fi_len, replace=True)[fi > 0.]

        if separable:
            fi = 0.

        rho = generate_parametrized_qs(qbits, a, c, fi)
        if local_randomization:
            rho = local_randomize_matrix(np.arange(qbits), rho, 2)

        if permute:
            qubits_perm = list(permutations(list(np.arange(qbits))))
            rand_perm = qubits_perm[np.random.randint(len(qubits_perm))]
            rho = permute_matrix(rand_perm, rho)
        
        save_dens_matrix_with_labels(qbits, f"dens{indx + i}", rho, "random parametrized", 'unknown', save_data_dir, separate_bipart=encoded, zero_neg= zero_neg, discord = discord)

    
                
# ENTANGLEMENT TRAIN SET 1/2 (SAME FOR ALL 3 DATASETS) - always verified in case of
def generate_pure_train_balanced(qubits, encoded, indx = 0, save_data_dir = 'train_balanced', examples_ratio = 1., max_num_ps = None, discord = False, biseparable = False, zero_neg = 'none'):
    args = {
        'qubits_num': qubits,
        'encoded': encoded,
        'start_index': indx,
        'save_data_dir': save_data_dir,
        'discord': discord,
        'start_index': indx,
    }
    generator = PureStatesGenerator()
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=int(examples_ratio*20000), specified_method=0)
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=int(examples_ratio*10000), specified_method= 1)
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=int(examples_ratio*10000), specified_method= 2)
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=int(examples_ratio*20000), specified_method = 3)
    args['start_index'] = generator.generate_random_haar_matrices(**args, examples=int(examples_ratio*20000))

    args['max_num_ps'] = max_num_ps
    generator = MixedDefStatesGenerator()
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, examples=int(examples_ratio*20000), num_pure_states=1, specified_method='kron_sep_haar')
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, examples=int(examples_ratio*40000), num_pure_states="random", specified_method=[0, 3], base_size=10000, zero_neg=zero_neg)
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, examples=int(examples_ratio*20000), num_pure_states="random", specified_method=0, base_size=10000)
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, examples=int(examples_ratio*30000), num_pure_states="random", specified_method=3, base_size=10000, zero_neg=zero_neg)
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, examples=int(examples_ratio*30000), num_pure_states="random", specified_method=4, base_size=10000, zero_neg=zero_neg)
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, examples=int(examples_ratio*20000), num_pure_states='random', specified_method= 'kron_sep_circ', mixing_mode='outer')
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, examples=int(examples_ratio*20000), num_pure_states= 'random', specified_method= 'kron_sep_circ', mixing_mode='inner')
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, examples=int(examples_ratio*20000), num_pure_states='random', specified_method= 'kron_sep_circ', mixing_mode='comb')
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, examples=int(examples_ratio*20000), num_pure_states='random', specified_method= 'kron_sep_haar', mixing_mode='outer')
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, examples=int(examples_ratio*20000), num_pure_states= 'random', specified_method= 'kron_sep_haar', mixing_mode='inner')
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, examples=int(examples_ratio*20000), num_pure_states='random', specified_method= 'kron_sep_haar', mixing_mode='comb')

    if biseparable:
        bisep_args = {
            'qubits_num': qubits,
            'encoded': encoded,
            'indx': args['start_index'],
            'save_data_dir': save_data_dir,
            'examples_ratio': examples_ratio,
            'max_num_ps': max_num_ps,
            'examples': 20000,
            'discord': discord
        }
        bisep_args['indx'] = generate_mixed_biseparable(**bisep_args, zero_neg=zero_neg)
        bisep_args['indx'] = generate_mixed_biseparable(**bisep_args, mixing_mode='inner', zero_neg=zero_neg)
        bisep_args['indx'] = generate_mixed_biseparable(**bisep_args, mixing_mode='comb', zero_neg=zero_neg)

        bisep_args['indx'] = generate_mixed_biseparable_haar(**bisep_args, nps = 1)
        bisep_args['indx'] = generate_mixed_biseparable_haar(**bisep_args)
        bisep_args['indx'] = generate_mixed_biseparable_haar(**bisep_args, mixing_mode = 'inner', zero_neg=zero_neg)
        bisep_args['indx'] = generate_mixed_biseparable_haar(**bisep_args, mixing_mode = 'comb', zero_neg=zero_neg)
        args['start_index'] = bisep_args['indx']
    return args['start_index']


# DISCORD PAPER TRAIN PURE SEPARABLE SET 1/2
# DISCORD PAPER VALIDATION SET 1/3 RATIO 0.1
def generate_pure_only_train_separable(qubits, encoded, indx = 0, save_data_dir = 'train_pure_separable', examples_ratio = 1., discord = False):
    args = {
        'qubits_num': qubits,
        'encoded': encoded,
        'start_index': indx,
        'save_data_dir': save_data_dir,
        'discord': discord,
    }
    pure_generator = PureStatesGenerator()
    args['start_index'] = pure_generator.generate_circuit_matrices(**args, specified_method=0, examples=int(examples_ratio*40000))

    mixed_generator = MixedDefStatesGenerator()
    args['start_index'] = mixed_generator.generate_multi_mixed_matrices(**args, examples=int(examples_ratio *40000), num_pure_states= 1, specified_method='kron_sep_circ', base_size=int(examples_ratio*20000))
    args['start_inxed'] = mixed_generator.generate_multi_mixed_matrices(**args, examples=int(examples_ratio*80000), num_pure_states=1, specified_method='kron_sep_haar', base_size=int(examples_ratio*40000))
    return args['start_index']


# DISCORD PAPER VALIDATION SET 2/3 RATIO 0.1
def generate_mixed_def_train_separable(qubits, encoded, indx = 0, save_data_dir = 'train_mixed_separable', examples_ratio = 1., discord = False, biseparable = True, max_num_ps = None):
    generator = MixedDefStatesGenerator()
    args = {
        'qubits_num': qubits,
        'save_data_dir': save_data_dir,
        'num_pure_states': 'random',
        'encoded': encoded,
        'discord': discord,
        'max_num_ps': max_num_ps,
        'start_index': indx,
    }

    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method=0, examples= int(examples_ratio*40000), base_size=int(examples_ratio*20000))
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method='kron_sep_circ', examples= int(examples_ratio*40000), base_size=int(examples_ratio*20000))
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method='kron_sep_haar', examples= int(examples_ratio*80000), base_size=int(examples_ratio*40000))

    if biseparable:
        args['start_index'] = generate_mixed_biseparable(qubits, encoded, args['start_index'], save_data_dir, examples=int(examples_ratio*80000), discord=discord)
        args['start_index'] = generate_mixed_biseparable_haar(qubits, encoded, args['start_index'], save_data_dir, examples=int(examples_ratio*80000), discord=discord)

    return args['start_index']


# DISCORD TRAIN PRODUCT EXTENSION SET 1/2
def generate_train_mixed_def_product(qubits, encoded, indx = 0, save_data_dir = 'train_mixed_separable', examples_ratio = 1., discord = False, max_num_ps = None):
    generator = MixedDefStatesGenerator()
    args = {
        'qubits_num': qubits,
        'save_data_dir': save_data_dir,
        'num_pure_states': 'random',
        'encoded': encoded,
        'discord': discord,
        'max_num_ps': max_num_ps,
        'start_index': indx,
        'examples': int(examples_ratio*40000),
        'mixing_mode': 'inner',
        'base_size': int(examples_ratio*20000),
    }
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method='kron_sep_circ')
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method='kron_sep_haar')
    return args['start_index']


def generate_mixed_biseparable(qubits, encoded, indx = 0, save_data_dir = 'biseparable', examples_ratio = 1., nps = 'random', max_num_ps = None, examples = 20000, with_permutations=True, discord = False, mixing_mode = 'outer', zero_neg = 'none'):
    generator = MixedDefStatesGenerator()
    return generator.generate_multi_mixed_matrices(
        qubits_num=qubits,
        examples = int(examples_ratio*examples),
        save_data_dir=save_data_dir,
        num_pure_states=nps,
        specified_method='biseparable',
        start_index=indx,
        encoded=encoded,
        max_num_ps=max_num_ps,
        zero_neg=zero_neg,
        fully_entangled=True,
        discord=discord,
        mixing_mode=mixing_mode,
        with_permutations=with_permutations
    )


def generate_mixed_biseparable_haar(qubits, encoded, indx = 0, save_data_dir = 'biseparable_haar', examples_ratio = 1., nps = 'random', max_num_ps = None, examples = 20000, with_permutations=True, discord = False, mixing_mode = 'outer', zero_neg='none'):
    generator = MixedDefStatesGenerator()
    return generator.generate_multi_mixed_matrices(
        qubits_num=qubits,
        examples = int(examples_ratio*examples),
        save_data_dir=save_data_dir,
        num_pure_states=nps,
        specified_method='biseparable_haar',
        start_index=indx,
        encoded=encoded,
        max_num_ps=max_num_ps,
        zero_neg=zero_neg,
        discord=discord,
        mixing_mode=mixing_mode,
        with_permutations=with_permutations
    )


# ENTANGLEMENT PAPER PURE TEST SET
# DISCORD PAPER PURE TEST SET
def generate_pure_test(qubits, encoded, indx = 0, save_data_dir = 'pure_test', discord=False):
    args = {
        'qubits_num': qubits,
        'save_data_dir': save_data_dir,
        'encoded': encoded,
        'discord': discord,
        'start_index': indx
    }
    pure_generator = PureStatesGenerator()
    args['start_index'] = pure_generator.generate_circuit_matrices(**args, examples=5000, specified_method= 0)
    args['start_index'] = pure_generator.generate_circuit_matrices(**args, examples=5000, specified_method= 3)
    args['start_index'] = pure_generator.generate_random_haar_matrices(**args, examples=10000)

    mixed_generator = MixedDefStatesGenerator()
    args['start_index'] = mixed_generator.generate_multi_mixed_matrices(**args, examples=5000, num_pure_states=1, specified_method='kron_sep_circ', base_size=5000)
    args['start_index'] = mixed_generator.generate_multi_mixed_matrices(**args, examples=5000, num_pure_states=1, specified_method='kron_sep_haar', base_size=5000)
    return args['start_index']


# ENTANGLEMENT TEST SET 1/2
# DISCORD TEST SET 1/3
def generate_mixed_test_def(qubits, encoded, indx = 0, save_data_dir='mixed_test_val', max_num_ps = None, discord = False, permute = False):
    generator = MixedDefStatesGenerator()
    args = {
        'qubits_num': qubits,
        'examples': 2500,
        'save_data_dir': save_data_dir,
        'start_index': indx,
        'with_permutations': permute,
        'num_pure_states': 'random',
        'base_size': 2500,
        'encoded': encoded,
        'max_num_ps': max_num_ps,
        'discord': discord
    }
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method='kron_sep_circ', mixing_mode='outer')
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method='kron_sep_circ', mixing_mode='inner')
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method='kron_sep_haar', mixing_mode='outer')
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method='kron_sep_haar', mixing_mode='inner')
    
    args['base_size'] = 5000
    args['examples'] = 5000
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method=0)
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method=3, zero_neg='zero_discord')
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method=4, zero_neg='zero_discord')

    return args['start_index']


# DISCORD TRAIN ZD EXTENSION RATIO 10 
# DISCORD TRAIN NON PRODUCT SET 1/2 RATIO 20
# DISCORD TEST SET 3/3 RATIO 1.5
def generate_non_product_zero_discord(qubits, encoded, indx = 0, save_data_dir = 'npzd_test', examples_ratio = 1., discord = False, max_num_ps = None):
    generator = MixedDefStatesGenerator()
    args = {
        'qubits_num': qubits, 
        'examples': int(examples_ratio*5000),
        'save_data_dir': save_data_dir,
        'base_size': int(examples_ratio*5000),
        'max_num_ps': max_num_ps,
        'encoded': encoded,
        'discord': discord,
        'start_index': indx,
    }
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, num_pure_states=2, specified_method='simple_non_product_zero_discord')
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, num_pure_states='random', specified_method='random_non_product_zero_discord')
    return args['start_index']


# DISCORD TRAIN SEP EXTENSION RATIO 6
# DISCORD TRAIN NON PRODUCT SET 2/2 RATIO 8
def generate_discordant_separable(qubits, encoded, indx = 0, save_data_dir = 'ds_test', examples_ratio = 1., discord = False, max_num_ps = None):
    generator = MixedDefStatesGenerator()
    args = {
        'qubits_num': qubits, 
        'save_data_dir': save_data_dir,
        'num_pure_states': 'random',
        'max_num_ps': max_num_ps,
        'encoded': encoded,
        'discord': discord,
        'start_index': indx,
    }
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method=0, examples=int(examples_ratio*10000), base_size=int(examples_ratio*10000))
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method='kron_sep_circ', examples=int(examples_ratio*5000), base_size=int(examples_ratio*5000), mixing_mode='outer')
    args['start_index'] = generator.generate_multi_mixed_matrices(**args, specified_method='kron_sep_haar', examples=int(examples_ratio*5000), base_size=int(examples_ratio*5000), mixing_mode='outer')
    return args['start_index']


def generate_pptes(qbits, encoded, indx = 0, save_data_dir = 'pptes', examples_ratio = 1., discord = False):
    generator = PPTESGenerator()
    args = {
        'qubits_num': qbits, 
        'examples': int(examples_ratio*10000),
        'with_permutations': True,
        'save_data_dir': save_data_dir,
        'randomization': True,
        'encoded': encoded,
        'discord': discord,
        'start_index': indx,
    }
    args['start_index'] = generator.generate_3qbits_pptes(**args, entanglement_class='Horodecki')
    args['start_index'] = generator.generate_3qbits_pptes(**args, entanglement_class='Acin')
    args['start_index'] = generator.generate_3qbits_pptes(**args, entanglement_class='BennetBE')
    return args['start_index']


# ENTANGLEMENT PPTES ACIN SET
def generate_acin(qbits, encoded, indx = 0, save_data_dir = 'pptes_acin', examples_ratio = 1., discord = False):
    generator = PPTESGenerator()
    args = {
        'qubits_num': qbits, 
        'examples': int(examples_ratio*10000),
        'with_permutations': True,
        'save_data_dir': save_data_dir,
        'randomization': True,
        'encoded': encoded,
        'discord': discord,
        'start_index': indx,
    }
    args['start_index'] = generator.generate_3qbits_pptes(**args, entanglement_class='Acin')
    return args['start_index']


# ENTANGLEMENT PPTES ACIN SET
def generate_extended_acin(qbits, encoded, indx = 0, save_data_dir = 'pptes_extended_acin', examples_ratio = 1., discord = False):
    generator = PPTESGenerator()
    args = {
        'qubits_num': qbits, 
        'examples': int(examples_ratio*10000),
        'with_permutations': True,
        'save_data_dir': save_data_dir,
        'randomization': True,
        'encoded': encoded,
        'discord': discord,
        'start_index': indx,
    }
    args['start_index'] = generator.generate_extended_3q_pptes(**args, entanglement_class='Acin')
    return args['start_index']


# ENTANGLEMENT PPTES HORODECKI SET
def generate_horodecki(qbits, encoded, indx = 0, save_data_dir = 'pptes_horodecki', examples_ratio = 1., discord = False):
    generator = PPTESGenerator()
    args = {
        'qubits_num': qbits, 
        'examples': int(examples_ratio*10000),
        'with_permutations': True,
        'save_data_dir': save_data_dir,
        'randomization': True,
        'encoded': encoded,
        'discord': discord,
        'start_index': indx,
    }
    args['start_index'] = generator.generate_3qbits_pptes(**args, entanglement_class='Horodecki')
    return args['start_index']


# ENTANGLEMENT PPTES BENNET SET
def generate_bennet(qbits, encoded, indx = 0, save_data_dir = 'pptes_bennet', examples_ratio = 1., discord = False):
    generator = PPTESGenerator()
    args = {
        'qubits_num': qbits, 
        'examples': int(examples_ratio*10000),
        'with_permutations': True,
        'save_data_dir': save_data_dir,
        'randomization': True,
        'encoded': encoded,
        'discord': discord,
        'start_index': indx,
    }
    args['start_index'] = generator.generate_3qbits_pptes(**args, entanglement_class='BennetBE')
    return args['start_index']


# ENTANGLEMENT PAPER TRAIN SETS 2/2
def generate_mixed_reduced_train_balanced(qubits, encoded, indx = 0, ppt = True, save_data_dir = 'train_balanced', examples_ratio = 1., zero_neg = 'incl', qubits_glob = 9, discord = False):
    args = {
        'qubits_num': qubits,
        'pure_state_qubits': qubits_glob,
        'encoded': encoded,
        'start_index': indx,
        'save_data_dir': save_data_dir,
        'discord': discord,
    }

    generator = MixedReducedStatesGenerator()
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=int(examples_ratio*40000), specified_method=ReducedMethods.RandomEntanglement, label_potent_ppt=ppt, zero_neg=zero_neg)
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=int(examples_ratio*5000), specified_method=ReducedMethods.Wstate)
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=int(examples_ratio*5000), specified_method=ReducedMethods.GHZstate)
    args['start_index'] = generator.generate_random_haar_matrices(**args, examples=int(examples_ratio*10000))
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=int(examples_ratio*20000), specified_method=ReducedMethods.Separable)
    return args['start_index']


# DISCORD PAPER VALIDATION SET 3/3
def generate_mixed_reduced_val(qubits, encoded, indx = 0, save_data_dir = 'train_separable', examples_ratio = 1., qubits_glob = 9, discord = False):
    args = {
        'qubits_num': qubits,
        'pure_state_qubits': qubits_glob,
        'encoded': encoded,
        'start_index': indx,
        'save_data_dir': save_data_dir,
        'discord': discord,
    }

    generator = MixedReducedStatesGenerator()
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=int(examples_ratio*4000), specified_method=ReducedMethods.Separable)
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=int(examples_ratio*4000), specified_method=ReducedMethods.CarrierCorrelatedNoEntanglement)
    args['start_index'] = generator.generate_random_haar_matrices(**args, examples=int(examples_ratio*4000), separable_only=True)
    return args['start_index']


# DISCORD TRAIN PRODUCT EXTENSION SET 2/2
def generate_mixed_reduced_train_product(qubits, encoded, indx = 0, save_data_dir = 'train_product', examples_ratio = 1., qubits_glob = 9, discord = False):
    args = {
        'qubits_num': qubits,
        'pure_state_qubits': qubits_glob,
        'encoded': encoded,
        'start_index': indx,
        'save_data_dir': save_data_dir,
        'discord': discord,
    }
    
    generator = MixedReducedStatesGenerator()
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=int(examples_ratio*40000), specified_method=ReducedMethods.Separable)

    return args['start_index']


# ENTANGLEMENT TEST SET 2/2
# Actual mixed test set generation method
def generate_mixed_reduced_test(qubits, encoded, indx = 0, qubits_glob = 9, save_data_dir= 'mixed_test_val', discord = False, permute = False):
    args = {
        'qubits_num': qubits,
        'pure_state_qubits': qubits_glob,
        'encoded': encoded,
        'start_index': indx,
        'save_data_dir': save_data_dir,
        'discord': discord,
        'with_permutations': permute
    }
    generator = MixedReducedStatesGenerator()
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=5000, specified_method=ReducedMethods.Separable)
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=5000, specified_method=ReducedMethods.RandomEntanglement, zero_neg='zero_discord')
    args['start_index'] = generator.generate_random_haar_matrices(**args, examples=5000, zero_neg='zero_discord')
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=5000, specified_method=ReducedMethods.CarrierCorrelated, zero_neg='incl')
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=5000, specified_method=ReducedMethods.DirectCorrelated, zero_neg='zero_discord')
    return args['start_index']


# DISCORD TEST SET 2/3
# Verified mixed test set split into (prod, non-prod-non-discord-sep, ent)
def generate_mixed_test_3class(qubits, encoded, indx = 0, qubits_glob = 9, save_data_dir= 'mixed_test_3c', discord = False, permute = False):
    args = {
        'qubits_num': qubits,
        'pure_state_qubits': qubits_glob,
        'encoded': encoded,
        'start_index': indx,
        'save_data_dir': save_data_dir,
        'discord': discord,
        'with_permutations': permute
    }

    generator = MixedReducedStatesGenerator()
    # Product states
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=2500, specified_method=ReducedMethods.Separable)
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=2500, specified_method=ReducedMethods.RandomEntanglement, zero_neg='zero_discord_only')
    args['start_index'] = generator.generate_random_haar_matrices(**args, examples=2500, zero_neg='zero_discord_only')
   
    # Non-product non-discord separable states
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=7500, specified_method=ReducedMethods.CarrierCorrelated, zero_neg='only')

    # Entangled states
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=5000, specified_method=ReducedMethods.RandomEntanglement, zero_neg='none')
    args['start_index'] = generator.generate_circuit_matrices(**args, examples=5000, specified_method=ReducedMethods.DirectCorrelated, zero_neg='none')
    return args['start_index']


# ENTANGLEMENT PAPER TRAIN SETS (WEAKLY LABELED - label_ppt = True, zero_neg = 'incl', NEGATIVITY LABELED - label_ppt = False, zero_neg = 'incl', NO PPTES - label_ppt = False, zero_neg = 'none')
# FOR VALIDATION THE SAME AS FOR NOPPTES with examples_ratio = 0.1
def generate_train_balanced(qbits, encoded, indx = 0, label_ppt = True, save_data_dir = 'train_balanced', examples_ratio = 1., max_num_ps = None, zero_neg = 'incl', qubits_glob = 9):
    new_indx = generate_pure_train_balanced(qbits, encoded, indx, save_data_dir, examples_ratio, max_num_ps)
    new_indx = generate_mixed_reduced_train_balanced(qbits, encoded, new_indx, ppt = label_ppt, save_data_dir = save_data_dir, examples_ratio = examples_ratio, zero_neg= zero_neg, qubits_glob=qubits_glob)
    return new_indx


# DISCORD PAPER FULL TRAIN SEPARABLE SET
def generate_full_train_separable(qbits, encoded, indx = 0, save_data_dir = 'train_separable', examples_ratio = 1., discord = False, qubits_glob = 9):
    # Pure separable
    indx = generate_pure_only_train_separable(qbits, encoded, indx = indx, save_data_dir = save_data_dir, examples_ratio = examples_ratio, discord = discord)
    indx = generate_parametrized(qbits, encoded, indx = indx, save_data_dir = save_data_dir, examples = int(examples_ratio*30000), discord = discord, separable = True, pure = True)
    # Product
    indx = generate_train_mixed_def_product(qbits, encoded, indx = indx, save_data_dir = save_data_dir, examples_ratio = examples_ratio, discord = discord)
    indx = generate_mixed_reduced_train_product(qbits, encoded, indx = indx, save_data_dir = save_data_dir, examples_ratio = examples_ratio, discord = discord, qubits_glob = qubits_glob)
    # Zero discord
    indx = generate_non_product_zero_discord(qbits, encoded, indx = indx, save_data_dir = save_data_dir, examples_ratio = 10*examples_ratio, discord = discord)
    # Mixed separable
    indx  = generate_discordant_separable(qbits, encoded, indx = indx, save_data_dir = save_data_dir, examples_ratio = 6*examples_ratio, discord = discord)
    return indx


# DISCORD PAPER TRAIN NON PRODUCT SEPARABLE SET
def generate_train_non_product_separable(qbits, encoded, indx = 0, save_data_dir = 'train_non_product_separable', examples_ratio = 1., discord = False):
    indx = generate_non_product_zero_discord(qbits, encoded, indx = indx, save_data_dir = save_data_dir, examples_ratio = 20*examples_ratio, discord = discord)
    indx = generate_discordant_separable(qbits, encoded, indx = indx, save_data_dir = save_data_dir, examples_ratio = 8*examples_ratio, discord = discord)
    return indx


# DISCORD PAPER VALIDATION SET WITH examples_ratio = 0.1
def generate_train_separable(qbits, encoded, indx = 0, save_data_dir = 'train_separable', examples_ratio = 1., discord = False, qubits_glob = 9):
    new_indx = generate_pure_only_train_separable(qbits, encoded, indx, save_data_dir, examples_ratio, discord)
    new_indx = generate_mixed_def_train_separable(qbits, encoded, new_indx, save_data_dir, examples_ratio, discord, biseparable=False)
    new_indx = generate_mixed_reduced_val(qbits, encoded, new_indx, save_data_dir, examples_ratio, qubits_glob, discord)
    return new_indx


# DISCORD PAPER TRAIN PRODUCT EXTENSION FOR TRAIN PURE SEPARABLE SET
def generate_train_mixed_product(qbits, encoded, indx = 0, save_data_dir = 'train_mixed_product', examples_ratio = 1., discord = False, qubits_glob = 9):
    new_indx = generate_train_mixed_def_product(qbits, encoded, indx, save_data_dir, examples_ratio, discord, biseparable=False)
    new_indx = generate_mixed_reduced_train_product(qbits, encoded, new_indx, save_data_dir, examples_ratio, qubits_glob, discord)
    return new_indx


# ENTANGLEMENT PAPER MIXED TEST SET
def generate_mixed_test_set(qbits, encoded, indx = 0, qubits_glob = 9, save_data_dir = 'mixed_test_val', max_num_ps = None, discord = False, permute = False):
    new_indx = generate_mixed_test_def(qbits, encoded, indx, save_data_dir, max_num_ps, discord, permute)
    new_indx = generate_mixed_reduced_test(qbits, encoded, new_indx, qubits_glob, save_data_dir, discord, permute)
    return new_indx


# DISCORD PAPER MIXED TEST SET
def generate_mixed_balanced_test_set(qbits, encoded, indx = 0, qubits_glob = 9, save_data_dir = 'mixed_test_bal', max_num_ps = None, discord = False, permute = False):
    new_indx = generate_mixed_test_def(qbits, encoded, indx, save_data_dir, max_num_ps, discord, permute)
    new_indx = generate_mixed_test_3class(qbits, encoded, new_indx, qubits_glob, save_data_dir, discord, permute)
    new_indx = generate_non_product_zero_discord(qbits, encoded, new_indx, save_data_dir, examples_ratio=1.5, discord=discord, max_num_ps=max_num_ps)
    return new_indx
