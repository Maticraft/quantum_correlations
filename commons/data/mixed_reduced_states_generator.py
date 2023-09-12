import random
from itertools import permutations
from enum import IntEnum

import numpy as np
from qiskit import *
from qiskit import Aer
from qiskit.quantum_info import DensityMatrix, partial_trace, random_statevector

from commons.data.savers import save_dens_matrix_with_labels
from commons.data.decorators import generate_with_assertion
from commons.data.circuit_ops import W_state, GHZ_state, local_randomization, random_entanglement, random_pair_entanglement, permute_matrix


class ReducedMethods(IntEnum):
    Wstate = 0
    GHZstate = 1
    RandomEntanglement = 2
    SimpleRandom = 210
    Separable = 3
    Biseparable = 4
    CarrierCorrelated = 5
    CarrierCorrelatedNoEntanglement = 51
    CarrierCorrelatedEntanglement = 52
    DirectCorrelated = 6


# Mixed states generator from trace of the global pure state
class MixedReducedStatesGenerator():
    def __init__(self, device = 'CPU'):
        self.backend = Aer.get_backend('statevector_simulator')
        if device == 'GPU':
            self.backend.set_options(device='GPU')

        self.num_qubits = None
        self.pure_qubits = None
        self.num_examples = None
        self.with_permutations = False

        self.qreg = None
        self.creg = None
        self.circuits = []


    def _initialize_generator(self, examples, num_qubits, pure_state_qubits, with_permutations):
        self.num_qubits = num_qubits
        self.pure_qubits = pure_state_qubits
        self.num_examples = examples

        self.with_permutations = with_permutations

        self.qreg = QuantumRegister(self.pure_qubits, 'q')
        self.creg = ClassicalRegister(self.pure_qubits, 'c')
        self.circuits = [QuantumCircuit(self.qreg, self.creg) for _ in range(examples)]


    def _mixed_state_from_Wstate(self, circuit, qubits_for_entanglement, random_gates):
        qbits = list(np.arange(self.pure_qubits))
        sample = random.sample(qbits, qubits_for_entanglement)

        W_state(self.qreg[sample], circuit)
        local_randomization(self.qreg[qbits], circuit, random_gates)

        return sample

    def _mixed_state_from_GHZ(self, circuit, qubits_for_entanglement, random_gates):
        qbits = list(np.arange(self.pure_qubits))
        sample = random.sample(qbits, qubits_for_entanglement)

        GHZ_state(self.qreg[sample], circuit)
        local_randomization(self.qreg[qbits], circuit, random_gates)

        return sample


    def _mixed_random_entangled_state(self, circuit, qubits_for_entanglement, random_gates):
        qbits = list(np.arange(self.pure_qubits))
        sample = random.sample(qbits, qubits_for_entanglement)

        local_randomization(self.qreg[sample], circuit, 1)
        random_entanglement(self.qreg[sample], circuit, -1)
        local_randomization(self.qreg[qbits], circuit, random_gates)

        return sample


    # introducing correlation in subsystem via carrier qubits from external system
    def _mixed_random_carrier_state(self, circuit, ent_qbits_in_sub, carriers, random_gates, entanglement_in_subsystem = False):
        qbits = list(np.arange(self.pure_qubits))
        sample_sub = random.sample(qbits, ent_qbits_in_sub)
        sample_carr = random.sample([q for q in qbits if q not in sample_sub], carriers)

        local_randomization(self.qreg[qbits], circuit, 1)

        if len(sample_carr) >= 2:
            random_entanglement(self.qreg[sample_carr], circuit, -1)

        for qbit in sample_sub:
            random_pair_entanglement([self.qreg[qbit], self.qreg[random.choice(sample_carr)]], circuit)

        local_randomization(self.qreg[qbits], circuit, random_gates)

        if entanglement_in_subsystem and len(sample_sub) >= 2:
            random_entanglement(self.qreg[sample_sub], circuit, len(sample_sub))
            local_randomization(self.qreg[sample_sub], circuit, random_gates)

        return sample_sub + sample_carr, sample_sub


    # direct correlation in subsystem
    def _mixed_random_dcs_state(self, circuit, ent_qbits_in_sub, extra_qubits_for_entanglement, random_gates):
        qbits = list(np.arange(self.pure_qubits))
        sample_sub = random.sample(qbits, ent_qbits_in_sub)
        sample_ext = random.sample([q for q in qbits if q not in sample_sub], extra_qubits_for_entanglement)

        local_randomization(self.qreg[qbits], circuit, 1)
        random_entanglement(self.qreg[sample_sub], circuit, -1)

        for i, qbit in enumerate(sample_ext):
            random_pair_entanglement([self.qreg[qbit], self.qreg[random.choice(sample_sub + sample_ext[:i])]], circuit)
        
        local_randomization(self.qreg[qbits], circuit, random_gates)

        return sample_sub + sample_ext, sample_sub


    @generate_with_assertion(batch_size=100)
    def generate_circuit_matrices(self, examples, qubits_num, pure_state_qubits, save_data_dir = None, random_gates = 2, specified_method = None, start_index = 0, encoded = True, label_potent_ppt = False, zero_neg = 'incl', discord = False, fully_entangled = False, with_permutations = False, num_near_zero_eigvals = None):
        # possible methods:
        # 0: W state
        # 1: GHZ state
        # 2: random entanglement
        # 3: separable
        # 4: biseparable
        # 5: carrier correlated state (cc state) with random entanglement in subsystem
        # 51: carrier correlated state without entanglement in subsystem
        # 52: carrier correlated state with entanglement in subsystem
        # 6: direct correlated in subsystem states (dcs state) 
        if specified_method not in list(ReducedMethods):
            raise ValueError("Invalid method!")

        global_info = {
            'save_data_dir': save_data_dir,
            'separate_bipart': encoded,
            'zero_neg': zero_neg,
            'discord': discord,     
            'num_near_zero_eigvals': num_near_zero_eigvals
        }

        self._initialize_generator(examples, qubits_num, pure_state_qubits, with_permutations)
        params = self._prepare_circuits(random_gates, specified_method, label_potent_ppt, fully_entangled)
        dens_matrices = self._execute_circuits()

        for i in range(len(dens_matrices)):
            reduced_ro, additional_info = self._calculate_reduced_matrix(specified_method, label_potent_ppt, fully_entangled, params, dens_matrices, i)
            additional_info.update(global_info)

            if self.with_permutations:
                reduced_ro, permutation = self._permute_matrix(reduced_ro)
                additional_info['not_ent_qbits'] = self._permute_qubit_ids(additional_info['not_ent_qbits'], permutation)

            save_dens_matrix_with_labels(self.num_qubits, f"dens{start_index + i}", reduced_ro, **additional_info)


    def _calculate_reduced_matrix(self, specified_method, label_potent_ppt, fully_entangled, params, dens_matrices, i):
        if specified_method in [ReducedMethods.Wstate, ReducedMethods.GHZstate, ReducedMethods.RandomEntanglement]:
            reduced_ro, info = self._calculate_reduced_matrix_for_simple_states(label_potent_ppt, fully_entangled, params, dens_matrices, i)
            
        elif specified_method == ReducedMethods.Separable:
            reduced_ro, info = self._calculate_reduced_matrix_for_separable_states(params, dens_matrices)

        elif specified_method == ReducedMethods.Biseparable:
            reduced_ro, info = self._calculate_reduced_matrix_for_biseparable_states(params, dens_matrices)

        elif specified_method in [ReducedMethods.CarrierCorrelated, ReducedMethods.CarrierCorrelatedEntanglement, ReducedMethods.CarrierCorrelatedNoEntanglement, ReducedMethods.DirectCorrelated]:
            reduced_ro, info = self._calculate_reduced_matrix_for_correlation_enforced_states(label_potent_ppt, params, dens_matrices, i)

        else:
            raise ValueError("Not recognized method!")

        return reduced_ro, info


    def _calculate_reduced_matrix_for_simple_states(self, label_potent_ppt, fully_entangled, params, dens_matrices, i):
        m, entangled_qbits = params[i]
        ro = dens_matrices[i]

        if fully_entangled:
            min_num_ent_qbits = self.num_qubits
        else:
            min_num_ent_qbits = max(1, self.num_qubits - self.pure_qubits + len(entangled_qbits))
        max_num_ent_qbits = min(self.num_qubits, len(entangled_qbits))

        entangled_qbits_in_subsystem = random.sample(entangled_qbits, random.randint(min_num_ent_qbits, max_num_ent_qbits))
        qbits_for_trace = [q for q in entangled_qbits if q not in entangled_qbits_in_subsystem]
                
        k = self.pure_qubits - self.num_qubits - len(qbits_for_trace)
        if k > 0:
            qbits_for_trace += random.sample([q for q in range(self.pure_qubits) if q not in entangled_qbits], k)

        not_entangled_qbits_inds = []

        if label_potent_ppt and m == ReducedMethods.RandomEntanglement and len(entangled_qbits_in_subsystem) >=2:
            pptes_flag = True
        else:
            pptes_flag = False

        if len(entangled_qbits_in_subsystem) < self.num_qubits:
            not_entangled_qbits = [q for q in range(self.pure_qubits) if (q not in qbits_for_trace) and (q not in entangled_qbits_in_subsystem)]
            qbits_in_subsystem = [q for q in range(self.pure_qubits) if q not in qbits_for_trace]
            not_entangled_qbits_inds = [qbits_in_subsystem.index(q) for q in not_entangled_qbits]  # TODO: check if it is correct

        reduced_ro = partial_trace(ro, qbits_for_trace)
        ent_qbits_str = str(len(entangled_qbits_in_subsystem)) + "/" + str(len(entangled_qbits))
        return reduced_ro, {
            'method': ReducedMethods(m).name,
            'entangled_qbits': ent_qbits_str,
            'ppt': pptes_flag,
            'not_ent_qbits': not_entangled_qbits_inds,          
        }


    def _calculate_reduced_matrix_for_separable_states(self, params, dens_matrices):
        rand_ind = random.randint(0, len(dens_matrices) - 1)
        m, entangled_qbits = params[rand_ind]
        ro = dens_matrices[rand_ind]
        ent_q = entangled_qbits[random.randint(0, len(entangled_qbits) - 1)]
        qbits_for_trace = [q for q in range(self.pure_qubits) if q != ent_q]
        reduced_ro = partial_trace(ro, qbits_for_trace).data

        for _ in range(1, self.num_qubits):
            rand_ind = random.randint(0, len(dens_matrices) - 1)
            m, entangled_qbits = params[rand_ind]
            ro = dens_matrices[rand_ind]
            ent_q = entangled_qbits[random.randint(0, len(entangled_qbits) - 1)]
            qbits_for_trace = [q for q in range(self.pure_qubits) if q != ent_q]
            reduced_ro = np.kron(reduced_ro, partial_trace(ro, qbits_for_trace).data) 
        reduced_ro = DensityMatrix(reduced_ro)
        ent_qbits_str = 'sep'

        pptes_flag = False
        not_entangled_qbits_inds = []
        return reduced_ro, {
            'method': ReducedMethods(m).name,
            'entangled_qbits': ent_qbits_str,
            'ppt': pptes_flag,
            'not_ent_qbits': not_entangled_qbits_inds,          
        }


    def _calculate_reduced_matrix_for_biseparable_states(self, params, dens_matrices):
        rand_ind = random.randint(0, len(dens_matrices) - 1)
        m, entangled_qbits = params[rand_ind]
        ro = dens_matrices[rand_ind]
        k = random.randint(1, min(len(entangled_qbits), self.num_qubits - 1))
        ent_q = random.sample(entangled_qbits, k)

        qbits_for_trace = [q for q in range(self.pure_qubits) if q not in ent_q]
        reduced_ro = partial_trace(ro, qbits_for_trace)
        reduced_ro_qubits = reduced_ro.data.shape[-1]

        rand_ind = random.randint(0, len(dens_matrices) - 1)
        m, entangled_qbits = params[rand_ind]
        while len(entangled_qbits) < (self.num_qubits - k + 1):
            rand_ind = random.randint(0, len(dens_matrices) - 1)
            m, entangled_qbits = params[rand_ind]

        ent_q = random.sample(entangled_qbits, self.num_qubits - k)
        qbits_for_trace = [q for q in range(self.pure_qubits) if q not in ent_q]
        reduced_ro = reduced_ro.tensor(partial_trace(ro, qbits_for_trace))

        ent_qbits_str = 'bisep'

        pptes_flag = False
        not_entangled_qbits_inds = [tuple([self.num_qubits - i for i in range(1, reduced_ro_qubits + 1)])]
        return reduced_ro, {
            'method': ReducedMethods(m).name,
            'entangled_qbits': ent_qbits_str,
            'ppt': pptes_flag,
            'not_ent_qbits': not_entangled_qbits_inds,          
        }


    def _calculate_reduced_matrix_for_correlation_enforced_states(self, label_potent_ppt, params, dens_matrices, i):
        m, entangled_qbits, entangled_qbits_in_subsystem = params[i]
        ro = dens_matrices[i]

        qbits_for_trace = [q for q in entangled_qbits if q not in entangled_qbits_in_subsystem]
                
        k = self.pure_qubits - self.num_qubits - len(qbits_for_trace)
        if k > 0:
            qbits_for_trace += random.sample([q for q in range(self.pure_qubits) if q not in entangled_qbits], k)

        not_entangled_qbits_inds = []

        if label_potent_ppt:
            pptes_flag = True
        else:
            pptes_flag = False

        if len(entangled_qbits_in_subsystem) < self.num_qubits:
            not_entangled_qbits = [q for q in range(self.pure_qubits) if (q not in qbits_for_trace) and (q not in entangled_qbits_in_subsystem)]
            qbits_in_subsystem = [q for q in range(self.pure_qubits) if q not in qbits_for_trace]
            not_entangled_qbits_inds = [qbits_in_subsystem.index(q) for q in not_entangled_qbits]

        reduced_ro = partial_trace(ro, qbits_for_trace)
        ent_qbits_str = str(len(entangled_qbits_in_subsystem)) + "/" + str(len(entangled_qbits))

        info = {
            'method': ReducedMethods(m).name,
            'entangled_qbits': ent_qbits_str,
            'ppt': pptes_flag,
            'not_ent_qbits': not_entangled_qbits_inds,          
        }
        return reduced_ro, info


    def _execute_circuits(self):
        executed = execute(self.circuits, self.backend).result()
        state_vectors = [executed.get_statevector(circ) for circ in self.circuits]
        dens_matrices = [DensityMatrix(vec) for vec in state_vectors]
        return dens_matrices


    def _prepare_circuits(self, random_gates, specified_method, label_potent_ppt, fully_entangled):
        params = [] 
        if specified_method == ReducedMethods.Biseparable and label_potent_ppt:
            raise NotImplementedError("Labeling potential PPTES not implemented for biseparable states.")

        for i in range(self.num_examples):
            m = self._determine_correct_pure_circuit_generation_method(specified_method)

            if m in [ReducedMethods.CarrierCorrelated, ReducedMethods.CarrierCorrelatedEntanglement, ReducedMethods.CarrierCorrelatedNoEntanglement]:
                entangled_qbits, entangled_qbits_in_subsystem = self._prepare_cc_circuit(random_gates, i, m)
                params.append((ReducedMethods.CarrierCorrelated, entangled_qbits, entangled_qbits_in_subsystem))

            elif m == ReducedMethods.DirectCorrelated:
                entangled_qbits, entangled_qbits_in_subsystem = self._prepare_dcs_circuit(random_gates, i)
                params.append((m, entangled_qbits, entangled_qbits_in_subsystem))

            else:
                entangled_qbits = self._prepare_simple_circuit(random_gates, fully_entangled, i, m)
                params.append((m, entangled_qbits))
        return params


    def _prepare_simple_circuit(self, random_gates, fully_entangled, i, m):
        methods = [
            self._mixed_state_from_Wstate,
            self._mixed_state_from_GHZ,
            self._mixed_random_entangled_state
        ]

        if fully_entangled:
            qbits_for_entanglement = random.choice(range(self.num_qubits + 1, self.pure_qubits + 1))
            return methods[m](self.circuits[i], qbits_for_entanglement, random_gates)
   
        entangled_qbits = []
        ent_repetitions = random.randint(1, self.num_qubits)
        for _ in range(ent_repetitions):
            qbits_for_entanglement = random.choice(range(2, self.pure_qubits + 1))
            entangled_qbits_j = methods[m](self.circuits[i], qbits_for_entanglement, random_gates)
            entangled_qbits = list(set(entangled_qbits + entangled_qbits_j))
        return entangled_qbits


    def _prepare_dcs_circuit(self, random_gates, i):
        ent_qbits_in_sub = random.randint(2, self.num_qubits)
        tot_ent_qbits = random.randint(1, self.pure_qubits - self.num_qubits)
        entangled_qbits, entangled_qbits_in_subsystem = self._mixed_random_dcs_state(self.circuits[i], ent_qbits_in_sub, tot_ent_qbits, random_gates)
        return entangled_qbits,entangled_qbits_in_subsystem


    def _prepare_cc_circuit(self, random_gates, i, m):
        ent_qbits_in_sub = random.randint(2, self.num_qubits)
        carriers = random.randint(1, self.pure_qubits - self.num_qubits)
        if m == ReducedMethods.CarrierCorrelated:
            ent_in_sub = bool(random.getrandbits(1))
        elif m == ReducedMethods.CarrierCorrelatedNoEntanglement:
            ent_in_sub = False
        elif m == ReducedMethods.CarrierCorrelatedEntanglement:
            ent_in_sub = True
        else:
            raise ValueError("Wrong method for carrier correlated state!")
        return self._mixed_random_carrier_state(self.circuits[i], ent_qbits_in_sub, carriers, random_gates, ent_in_sub)


    def _determine_correct_pure_circuit_generation_method(self, specified_method):
        if specified_method == ReducedMethods.SimpleRandom:
            return random.choice([ReducedMethods.Wstate, ReducedMethods.GHZstate, ReducedMethods.RandomEntanglement])
        elif specified_method == ReducedMethods.Separable or specified_method == ReducedMethods.Biseparable:
            return ReducedMethods.RandomEntanglement
        else:
            return specified_method

    
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
    

    # random from Haar metrics
    @generate_with_assertion(batch_size=100)
    def generate_random_haar_matrices(self, examples, qubits_num, pure_state_qubits, save_data_dir = None, start_index = 0, encoded = True, label_potent_ppt = False, zero_neg = 'incl', discord = False, separable_only = False, with_permutations = False):
        self._initialize_generator(examples, qubits_num, pure_state_qubits, with_permutations)

        not_ent_qbits = []
        dummy_indx = 0
        return_matrices = []

        for i in range(self.num_examples):
            num_pure_qubits = np.random.randint(self.num_qubits + 1, self.pure_qubits + 1)
            dim = 2**num_pure_qubits
            state = random_statevector(dim)
            matrix = DensityMatrix(state)

            if not separable_only:
                k = random.randint(num_pure_qubits - self.num_qubits, num_pure_qubits - 1)
            else:
                k = num_pure_qubits - 1
            qubits_for_trace = random.sample(list(range(num_pure_qubits)), k)
            ro = partial_trace(matrix, qubits_for_trace)
            ro_qb_num = num_pure_qubits - k # current number of qubits in ro

            while ro_qb_num < self.num_qubits:
                new_n = random.randint(self.num_qubits + 1, self.pure_qubits)
                new_dim = 2**new_n
                new_ro_glob = DensityMatrix(random_statevector(new_dim))
                if not separable_only:
                    new_k = random.randint(new_n - self.num_qubits + ro_qb_num, new_n - 1)
                else:
                    new_k = new_n - 1
                qubits_for_trace = random.sample(list(range(new_n)), new_k)
                new_ro = partial_trace(new_ro_glob, qubits_for_trace)
                ro = ro.tensor(new_ro)
                ro_qb_num += new_n - new_k 

                not_ent_qbits.append(dummy_indx) # just to increase the number of allowed zero-negativity bipartitions
                dummy_indx += 1

            if self.with_permutations:
                ro, _ = self._permute_matrix(ro)

            if save_data_dir:
                save_dens_matrix_with_labels(self.num_qubits, f"dens{start_index + i}", ro, f"random_mixed", 'random', save_data_dir, ppt = label_potent_ppt, separate_bipart=encoded, not_ent_qbits= not_ent_qbits, zero_neg= zero_neg, discord = discord)
            else:
                return_matrices.append(ro)

        if not save_data_dir:
            return return_matrices
