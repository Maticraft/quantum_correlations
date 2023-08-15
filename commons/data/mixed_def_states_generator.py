from itertools import permutations
import random

import numpy as np
from qiskit import Aer
from qiskit.quantum_info import DensityMatrix

from commons.data.savers import save_dens_matrix_with_labels
from commons.data.decorators import generate_with_assertion
from commons.data.circuit_ops import permute_matrix
from commons.data.pure_states_generator import PureStatesGenerator


# Generator of the mixed states based on the verified pure states
class MixedDefStatesGenerator():
    def __init__(self, device='CPU'):
        self.num_qubits = None
        self.with_permutations = False
        self.num_examples = None  
        self.pure_save_data_dir = None

        self.pure_gen = PureStatesGenerator(device=device)

    
    def _initialize_generator(self, examples, num_qubits, with_permutations):
        self.num_qubits = num_qubits
        self.num_examples = examples
        self.with_permutations = with_permutations


    @generate_with_assertion(1000)
    def generate_matrices(self, qubits_num, examples, save_data_dir = None, specified_method = None, start_index = 0, encoded = True, label_potent_ppt = False, with_permutations = False):
        self._initialize_generator(examples, qubits_num, with_permutations)
        d = 2**self.num_qubits

        return_matrices = []
        for i in range(self.num_examples):
            if specified_method == None:
                m = random.choice(range(5))
            else:
                m = specified_method

            ps = random.randint(2, d)
            ro = self._generate_single_mixed_matrix(ps, m)

            if m > 0 and label_potent_ppt:
                ppt_flag = True
            else:
                ppt_flag = False

            if self.with_permutations:
                ro, _ = self._permute_matrix(ro)

            if save_data_dir:
                save_dens_matrix_with_labels(self.num_qubits, f"dens{start_index + i}", ro, f"mixed from pure {m}", "unknown", save_data_dir, ppt = ppt_flag, separate_bipart= encoded)
            else:
                return_matrices.append(ro)
        if not save_data_dir:
            return return_matrices


    # Generate a single density matrix which comes from mixing pure states of a given type e.g. only random circuit generated states
    def _generate_single_mixed_matrix(self, num_pure_states, method, probabilities = 'random'):     
        if method < 4:
            ros = self.pure_gen.generate_circuit_matrices(self.num_qubits, examples = num_pure_states, save_data_dir = self.pure_save_data_dir, specified_method = method, return_matrices=True)
        elif method == 4:
            ros = self.pure_gen.generate_random_haar_matrices(self.num_qubits, examples = num_pure_states, save_data_dir = self.pure_save_data_dir, return_matrices=True)
        else:
            raise ValueError('Wrong method: {}'.format(method))

        dms = [ro.data for ro in ros]
        if probabilities == 'random':
            probs = np.random.uniform(size=num_pure_states)
            probs /= np.sum(probs)
            probs = np.expand_dims(probs, axis=(2,1))
        else:
            probs = np.expand_dims(probabilities, axis=(2,1))

        dm = np.sum(probs * dms, axis=0)

        return DensityMatrix(dm)


    def _permute_matrix(self, ro):
        qubits_perm = list(permutations(list(np.arange(self.num_qubits))))
        rand_perm = qubits_perm[np.random.randint(len(qubits_perm))]
        ro = permute_matrix(rand_perm, ro)
        return ro, rand_perm


    def _permute_qubit_ids(self, qubits_ids, permutation):
        permuted_qubit_ids = []
        for q in qubits_ids:
            if type(q) == int:
                new_q = permutation.index(q)
            else:
                new_q = tuple([permutation.index(qb) for qb in q])
            permuted_qubit_ids.append(new_q)
        return permuted_qubit_ids


    # More effective multi matrices generating method, which bases on some number of pregenerated pure states
    @generate_with_assertion(1000)
    def generate_multi_mixed_matrices(self, qubits_num, examples, num_pure_states, save_data_dir = None, specified_method = None, base_size = None, start_index = 0, encoded = True, label_potent_ppt = False, zero_neg = 'incl', fully_entangled = False, max_num_ps = None, discord = False, mixing_mode = 'outer', with_permutations = False):
        self._initialize_generator(examples, qubits_num, with_permutations)
        if base_size is None:
            base_size = examples
        not_ent_qbits, dms, dms2 = self._generate_pure_states(specified_method, base_size, fully_entangled)

        if num_pure_states == 'random':
            mixed_dms, npss = self._generate_mixed_density_matrices_from_random_states(specified_method, max_num_ps, mixing_mode, dms, dms2)
        elif type(num_pure_states) == int:
            mixed_dms = self._generate_mixed_density_matrices_from_constant_states(num_pure_states, specified_method, mixing_mode, dms, dms2)
        else:
            raise ValueError('Wrong number of pure states: {}'.format(num_pure_states))

        return_matrices = []
        for i in range(len(mixed_dms)):
            if num_pure_states == 'random':
                nps = npss[i]
            else:
                nps = num_pure_states
            ro = DensityMatrix(mixed_dms[i])
            
            if self.with_permutations:
                ro, permutation = self._permute_matrix(ro)
                not_ent_qbits = self._permute_qubit_ids(not_ent_qbits, permutation)

            if save_data_dir:
                if specified_method == 'simple_non_product_zero_discord' or specified_method == 'random_non_product_zero_discord':
                    save_dens_matrix_with_labels(self.num_qubits, f"dens{start_index + i}", ro, "random mixed from pure", "{}".format(nps), save_data_dir, ppt= label_potent_ppt, separate_bipart=encoded, zero_neg= zero_neg, not_ent_qbits=not_ent_qbits, discord = discord, trace_reconstruction=True)
                else:
                    save_dens_matrix_with_labels(self.num_qubits, f"dens{start_index + i}", ro, "random mixed from pure", "{}".format(nps), save_data_dir, ppt= label_potent_ppt, separate_bipart=encoded, zero_neg= zero_neg, not_ent_qbits=not_ent_qbits, discord = discord)
            else:
                return_matrices.append(ro)
        if not save_data_dir:
            return return_matrices


    def _generate_pure_states(self, specified_method, base_size, fully_entangled):
        not_ent_qbits = []
        dms, dms2 = [], []
        
        if specified_method == None:
            dms = self._generate_all_possible_pure_states(base_size, fully_entangled)

        elif type(specified_method) == list:
            dms = self._generate_pure_states_from_method_list(specified_method, base_size, fully_entangled)

        elif type(specified_method) == int:
            dms = self._generate_pure_states_from_method_int(specified_method, base_size, fully_entangled)
        
        elif specified_method == 'kron_sep_circ':
            not_ent_qbits += list(np.arange(self.num_qubits))
            dms = self._generate_single_qubit_circuit_pure_states(base_size)
        
        elif specified_method == 'kron_sep_haar':
            not_ent_qbits += list(np.arange(self.num_qubits))
            dms = self._generate_single_qubit_haar_pure_states(base_size)

        elif specified_method == 'biseparable': 
            dms, dms2 = self._generate_circuit_pure_states_pairs(base_size, fully_entangled)
            dms_qubits = int(np.log2(dms.shape[-1]))
            not_ent_qbits.append(tuple([self.num_qubits - i for i in range(1, dms_qubits + 1)]))

        elif specified_method == 'biseparable_haar':
            dms, dms2 = self._generate_haar_pure_states_pairs(base_size)
            dms_qubits = int(np.log2(dms.shape[-1]))
            not_ent_qbits.append(tuple([self.num_qubits - i for i in range(1, dms_qubits + 1)]))

        elif specified_method == 'simple_non_product_zero_discord':
            not_ent_qbits += list(np.arange(self.num_qubits))
            dms, dms2 = self._generate_simple_orthogonal_pairs()

        elif specified_method == 'random_non_product_zero_discord':
            not_ent_qbits += list(np.arange(self.num_qubits))
            dms = self._generate_random_orthogonal_pairs(base_size)

        else:
            raise Exception("Unknown method")
        return not_ent_qbits, dms, dms2


    def _generate_mixed_density_matrices_from_random_states(self, specified_method, max_num_ps, mixing_mode, dms, dms2):
        if max_num_ps == None:
            max_num_ps = 2*(2**self.num_qubits)
        mixed_dms = []
        npss = []
        indices = np.arange(len(dms))
        for i in range(self.num_examples):
            nps = random.randint(2, max_num_ps)
            npss.append(nps)
            probs = np.random.uniform(size= nps)
            probs /= np.sum(probs)
            probs = np.expand_dims(probs, axis = (2,1))

            if specified_method == 'kron_sep_circ' or specified_method == "kron_sep_haar":
                dm = self._generate_single_mixed_kron_separable_state(mixing_mode, dms, probs, nps, indices)
                mixed_dms.append(dm)

            elif specified_method == 'biseparable' or specified_method == 'biseparable_haar':
                dm = self._generate_single_mixed_biseparable_state(mixing_mode, dms, dms2, probs, nps, indices)
                mixed_dms.append(dm)
                
            elif specified_method == 'simple_non_product_zero_discord':
                raise Exception("simple_non_product_zero_discord method is only supported for number of pure states = 2")

            elif specified_method == 'random_non_product_zero_discord':
                dm = self._generate_single_mixed_random_npzd_state(dms, probs, nps, np.arange(len(dms // 2)))
                mixed_dms.append(dm)

            else:
                basis_states = np.random.choice(indices, size = nps)
                mixed_dms.append(np.sum(probs * dms[basis_states], axis = 0))

        mixed_dms = np.array(mixed_dms)
        return mixed_dms, npss


    def _generate_mixed_density_matrices_from_constant_states(self, num_pure_states, specified_method, mixing_mode, dms, dms2):
        nps = num_pure_states
        probs = np.random.uniform(size=(self.num_examples, nps))
        probs /= np.sum(probs, axis= 1, keepdims=True)
        probs = np.expand_dims(probs, axis = (2,3))
        indices = np.arange(len(dms))

        if specified_method == 'kron_sep_circ' or specified_method == "kron_sep_haar":
            mixed_dms = self._generate_mixed_kron_sep_states(num_pure_states, mixing_mode, dms, probs, indices)

        elif specified_method == 'biseparable' or specified_method == 'biseparable_haar':
            mixed_dms = self._generate_mixed_biseparable_states(num_pure_states, mixing_mode, dms, dms2, probs, indices)

        elif specified_method == 'simple_non_product_zero_discord':
            assert(num_pure_states == 2), "simple_non_product_zero_discord method is only supported for number of pure states = 2"
            mixed_dms = probs[:, 0] * dms + probs[:, 1] * dms2

        elif specified_method == 'random_non_product_zero_discord':
            mixed_dms = self._generate_mixed_random_npzd_states(dms, probs, nps, np.arange(len(dms) // 2))

        else:
            basis_states = np.random.choice(indices, size = (self.num_examples, nps))
            mixed_dms = np.sum(probs * dms[basis_states], axis = 1)
        return mixed_dms


    def _generate_single_mixed_kron_separable_state(self, mixing_mode, dms, probs, nps, indices):
        d = 2**self.num_qubits
        basis_states = np.random.choice(indices, size = (nps, self.num_qubits))
        if nps > 1:
            probs = np.squeeze(probs)
        else:
            probs = np.squeeze(probs, axis=(2, 1))
        dm = np.zeros((d, d), dtype=np.complex128)

        if mixing_mode == 'outer':
            for j in range(nps):
                dm_j = dms[basis_states[j, 0]]
                for k in range(1, self.num_qubits):
                    dm_jk = dms[basis_states[j, k]]
                    dm_j = np.kron(dm_j, dm_jk)
                dm += probs[j]*dm_j

        elif mixing_mode == 'inner':
            dm_n = probs[0]*dms[basis_states[0, 0]]
            for j in range(1, nps):
                dm_n += probs[j]*dms[basis_states[j, 0]]

            for k in range(1, self.num_qubits):
                dm_k = probs[0]*dms[basis_states[0, k]]
                for j in range(1, nps):
                    dm_k += probs[j]*dms[basis_states[j, k]]

                dm_n = np.kron(dm_n, dm_k)
            dm = dm_n
  
        else:
            nps2 = nps
            probs2 = np.random.uniform(size= nps2)
            probs2 /= np.sum(probs2)
            basis_states2 = np.random.choice(indices, size = (nps, nps2, self.num_qubits))

            for j in range(nps):
                dm_j = probs2[0]*dms[basis_states2[j, 0, 0]]
                for l in range(1, nps2):
                    dm_j += probs2[l]*dms[basis_states2[j, l, 0]]
                for k in range(1, self.num_qubits):
                    dm_jk = probs2[0]*dms[basis_states2[j, 0, k]]
                    for l in range(1, nps2):
                        dm_jk += probs2[l]*dms[basis_states2[j, l, k]]
                    dm_j = np.kron(dm_j, dm_jk)
                dm += probs[j]*dm_j

        return dm


    def _generate_single_mixed_biseparable_state(self, mixing_mode, dms, dms2, probs, nps, indices):
        d = 2**self.num_qubits
        basis_states = np.random.choice(indices, size = nps)
        if nps > 1:
            probs = np.squeeze(probs)
        else:
            probs = np.squeeze(probs, axis=(2, 1))
        dm = np.zeros((d,d), dtype = np.complex128)

        if mixing_mode == 'outer':
            for j in range(nps):
                dm_j = np.kron(dms[basis_states[j]], dms2[basis_states[j]])
                dm += probs[j]*dm_j

        elif mixing_mode == 'inner':
            dm_n1 = probs[0]*dms[basis_states[0]]
            dm_n2 = probs[0]*dms2[basis_states[0]]

            for j in range(1, nps):
                dm_n1 += probs[j]*dms[basis_states[j]]
                dm_n2 += probs[j]*dms2[basis_states[j]]

            dm = np.kron(dm_n1, dm_n2)
  
        else:
            nps2 = nps
            probs2 = np.random.uniform(size= nps2)
            probs2 /= np.sum(probs2)
            basis_states2 = np.random.choice(indices, size = (nps, nps2))

            for j in range(nps):
                dm_j1 = probs2[0]*dms[basis_states2[j, 0]]
                dm_j2 = probs2[0]*dms2[basis_states2[j, 0]]

                for l in range(1, nps2):
                    dm_j1 += probs2[l]*dms[basis_states2[j, l]]
                    dm_j2 += probs2[l]*dms2[basis_states2[j, l]]

                dm_j = np.kron(dm_j1, dm_j2)
                dm += probs[j]*dm_j
        return dm


    def _generate_single_mixed_random_npzd_state(self, dms, probs, nps, indices):
        d = 2**self.num_qubits
        basis_states = np.random.choice(indices, size = self.num_qubits)
        vectors_inds = np.random.randint(2, size = nps)
        probs = np.squeeze(probs, axis=(2, 1))

        dm = np.zeros((d, d), dtype=np.complex128)

        for j in range(nps):
            dm_j = dms[vectors_inds[j]*len(indices) + basis_states[0]]
            for k in range(1, self.num_qubits):
                dm_jk = dms[vectors_inds[j]*len(indices) + basis_states[k]]
                dm_j = np.kron(dm_j, dm_jk)
            dm += probs[j]*dm_j
        return dm


    def _generate_mixed_kron_sep_states(self, num_pure_states, mixing_mode, dms, probs, indices):
        nps = num_pure_states
        # if nps > 1:
        #     probs = np.squeeze(probs)
        # else:
        #     probs = np.squeeze(probs, axis=(2, 3))
        mixed_dms = []
        for i in range(self.num_examples):
            dm = self._generate_single_mixed_kron_separable_state(mixing_mode, dms, probs[i], nps, indices)
            mixed_dms.append(dm)

        mixed_dms = np.array(mixed_dms)
        return mixed_dms


    def _generate_mixed_biseparable_states(self, num_pure_states, mixing_mode, dms, dms2, probs, indices):
        nps = num_pure_states
        # if nps > 1:
        #     probs = np.squeeze(probs)
        # else:
        #     probs = np.squeeze(probs, axis=(2, 3))
        mixed_dms = []
        for i in range(self.num_examples):
            dm = self._generate_single_mixed_biseparable_state(mixing_mode, dms, dms2, probs[i], nps, indices)

            mixed_dms.append(dm)
        mixed_dms = np.array(mixed_dms)
        return mixed_dms


    def _generate_mixed_random_npzd_states(self, dms, probs, nps, indices):
        vectors_inds = np.random.randint(2, size = (self.num_examples, nps))
        basis_states = np.random.choice(indices, size = (self.num_examples, self.num_qubits))
        if nps > 1:
            probs = np.squeeze(probs)
        else:
            probs = np.squeeze(probs, axis=(2, 3))
        d = 2**self.num_qubits
        mixed_dms = []
        for i in range(self.num_examples):
            dm = np.zeros((d, d), dtype=np.complex128)

            for j in range(nps):
                dm_j = dms[vectors_inds[i, j]*len(indices) + basis_states[i, 0]]
                for k in range(1, self.num_qubits):
                    dm_jk = dms[vectors_inds[i, j]*len(indices) + basis_states[i, k]]
                    dm_j = np.kron(dm_j, dm_jk)
                dm += probs[i, j]*dm_j
                
            mixed_dms.append(dm)
        return mixed_dms


    def _generate_pure_states_from_method_int(self, specified_method, base_size, fully_entangled):
        if specified_method == 4:
            ros = self.pure_gen.generate_random_haar_matrices(self.num_qubits, base_size, self.pure_save_data_dir, return_matrices=True)
            dms = np.array([ro.data for ro in ros])
        else:
            ros = self.pure_gen.generate_circuit_matrices(self.num_qubits, examples= base_size, save_data_dir= self.pure_save_data_dir, specified_method = specified_method, fully_entangled=fully_entangled)
            dms = np.array([ro.data for ro in ros])
        return dms


    def _generate_haar_pure_states_pairs(self, base_size):
        num_q1 = random.randint(1, self.num_qubits // 2)
        num_q2 = self.num_qubits - num_q1
            
        ros1 = self.pure_gen.generate_random_haar_matrices(num_q1, base_size, self.pure_save_data_dir, return_matrices=True)
        dms1 = np.array([ro.data for ro in ros1])
        ros2 = self.pure_gen.generate_random_haar_matrices(num_q2, base_size, self.pure_save_data_dir, return_matrices=True)
        dms2 = np.array([ro.data for ro in ros2])
        return dms1, dms2


    def _generate_circuit_pure_states_pairs(self, base_size, fully_entangled):
        num_q1 = random.randint(1, self.num_qubits // 2)
        num_q2 = self.num_qubits - num_q1
        if num_q1 >= 2:
            m1 = 3
        else:
            m1 = 0
        if num_q2 >= 2:
            m2 = 3
        else:
            m2 = 0
        ros1 = self.pure_gen.generate_circuit_matrices(num_q1, base_size, self.pure_save_data_dir, specified_method=m1, fully_entangled=fully_entangled, return_matrices=True)
        dms1 = np.array([ro.data for ro in ros1])
        ros2 = self.pure_gen.generate_circuit_matrices(num_q2, base_size, self.pure_save_data_dir, specified_method=m2, fully_entangled=fully_entangled, return_matrices=True)
        dms2 = np.array([ro.data for ro in ros2])
        return dms1, dms2


    def _generate_single_qubit_haar_pure_states(self, base_size):
        ros = self.pure_gen.generate_random_haar_matrices(1, self.num_qubits*base_size, self.pure_save_data_dir, return_matrices=True)
        dms = np.array([ro.data for ro in ros])
        return dms


    def _generate_single_qubit_circuit_pure_states(self, base_size):
        ros = self.pure_gen.generate_circuit_matrices(1, self.num_qubits*base_size, self.pure_save_data_dir, specified_method=0, return_matrices=True)
        dms = np.array([ro.data for ro in ros])
        return dms


    def _generate_pure_states_from_method_list(self, specified_method, base_size, fully_entangled):
        dms = []
        for method in specified_method:
            if method == 4:
                ros = self.pure_gen.generate_random_haar_matrices(self.num_qubits, base_size // len(specified_method), self.pure_save_data_dir, return_matrices=True)
            else:
                ros = self.pure_gen.generate_circuit_matrices(self.num_qubits, base_size // len(specified_method), self.pure_save_data_dir, specified_method = method, fully_entangled = fully_entangled, return_matrices=True)
            dms += [ro.data for ro in ros]
        dms = np.array(dms)
        return dms


    def _generate_all_possible_pure_states(self, base_size, fully_entangled):
        ros_circ = self.pure_gen.generate_circuit_matrices(self.num_qubits, 3 * base_size // 4, self.pure_save_data_dir, fully_entangled=fully_entangled, return_matrices=True)
        ros_haar = self.pure_gen.generate_random_haar_matrices(self.num_qubits, base_size // 4, self.pure_save_data_dir) 
        dms = np.array([ro.data for ro in ros_circ] + [ro.data for ro  in ros_haar])
        return dms
    

    def _generate_simple_orthogonal_pairs(self):
        ros, ros2 = self.pure_gen.generate_random_separable_orthogonal_pairs(self.num_qubits, 2*self.num_examples)
        dms, dms2 = np.array([ro.data for ro in ros]), np.array([ro.data for ro in ros2])
        return dms,dms2


    def _generate_random_orthogonal_pairs(self, base_size):
        ros, ros2 = self.pure_gen.generate_random_separable_orthogonal_pairs(1, self.num_qubits*base_size)
        dms, dms2 = np.array([ro.data for ro in ros]), np.array([ro.data for ro in ros2])
        dms = np.concatenate((dms, dms2), axis=0)
        return dms
