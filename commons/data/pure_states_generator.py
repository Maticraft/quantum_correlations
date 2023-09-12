from itertools import permutations
import random

import numpy as np
from qiskit import *
from qiskit import Aer
from qiskit.quantum_info import DensityMatrix, random_statevector
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.providers.aer.noise import NoiseModel

from commons.data.savers import save_dens_matrix_with_labels
from commons.data.decorators import generate_with_assertion
from commons.data.circuit_ops import W_state, GHZ_state, local_randomization, random_entanglement, permute, permute_matrix, multiple_local_randomization


class PureStatesGenerator:
    def __init__(self, device='CPU'):
        self.backend = Aer.get_backend('statevector_simulator')
        if device == 'GPU':
            self.backend.set_options(device='GPU')

        self.methods = [
            self._generate_ps_state,
            self._generate_state_from_Wstate,
            self._generate_state_from_GHZ,
            self._generate_random_state
        ]

        self.num_qubits = None
        self.num_examples = None
        self.qreg = None
        self.creg = None
        self.circuits = [] 
        self.with_permutations = False
        self.num_permutations = 1


    def _initialize_generator(self, examples, num_qubits, with_permutations):
        self.num_qubits = num_qubits
        self.num_examples = examples
        self.qreg = QuantumRegister(num_qubits, 'q')
        self.creg = ClassicalRegister(num_qubits, 'c')
        self.circuits = [QuantumCircuit(self.qreg, self.creg) for ex in range(examples)]
        self.with_permutations = with_permutations
        self.num_permutations = len(list(permutations(range(num_qubits)))) if with_permutations else 1


    def _noise(self):
        # T1 and T2 values for qubits 0-3
        T1s = list(np.random.randint(10000, 25000, size = self.num_qubits))
        T2s = [random.randint(t1 // 2, 2*t1) for t1 in T1s]

        # Instruction times (in nanoseconds)
        time_u1 = 0   # virtual gate
        time_u2 = 50  # (single X90 pulse)
        time_u3 = 100 # (two X90 pulses)
        time_cx = 300
        time_reset = 1000  # 1 microsecond
        time_measure = 1000 # 1 microsecond

        # QuantumError objects
        errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                        for t1, t2 in zip(T1s, T2s)]
        errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                        for t1, t2 in zip(T1s, T2s)]
        errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
                    for t1, t2 in zip(T1s, T2s)]
        errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
                    for t1, t2 in zip(T1s, T2s)]
        errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
                    for t1, t2 in zip(T1s, T2s)]
        errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                    thermal_relaxation_error(t1b, t2b, time_cx))
                    for t1a, t2a in zip(T1s, T2s)]
                    for t1b, t2b in zip(T1s, T2s)]

        # Add errors to noise model
        noise_thermal = NoiseModel()

        for j in range(self.num_qubits):
            noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
            noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
            noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
            noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
            noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
            for k in range(self.num_qubits):
                noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])
        
        return noise_thermal


    def _generate_state_from_Wstate(self, circuit, qubits_for_entanglement, random_gates):
        qbits = list(np.arange(self.num_qubits))
        sample = random.sample(qbits, qubits_for_entanglement)
        W_state(self.qreg[sample], circuit)
        local_randomization(self.qreg[qbits], circuit, random_gates)


    def _generate_state_from_GHZ(self, circuit, qubits_for_entanglement, random_gates):
        qbits = list(np.arange(self.num_qubits))
        sample = random.sample(qbits, qubits_for_entanglement)
        GHZ_state(self.qreg[sample], circuit)
        local_randomization(self.qreg[qbits], circuit, random_gates)


    def _generate_ps_state(self, circuit, random_gates):
        qbits = list(np.arange(self.num_qubits))
        local_randomization(self.qreg[qbits], circuit, random_gates)


    def _generate_random_state(self, circuit, qubits_for_entanglement, random_gates, mode = 'random_entanglement'):
        qbits = list(np.arange(self.num_qubits))
        sample = random.sample(qbits, qubits_for_entanglement)
        local_randomization(self.qreg[sample], circuit, 1)
        if mode == 'random_entanglement':
            random_entanglement(self.qreg[sample], circuit, -1, 'random')
        elif mode == 'full_entanglement':
            random_entanglement(self.qreg[sample], circuit, 2*self.num_qubits, 'all')
        else:
            raise ValueError("Wrong mode!")
        local_randomization(self.qreg[qbits], circuit, random_gates)


    def _generate_random_ps_orthogonal_pair(self, circuits_pair, random_gates):
        qbits = list(np.arange(self.num_qubits))
        circuits_pair[1].x(qbits)
        multiple_local_randomization(qbits, circuits_pair, random_gates, preserve_orthogonality=True)


    @generate_with_assertion(1000)
    def generate_circuit_matrices(self, qubits_num, examples, save_data_dir = None, random_gates = 2, specified_method = None, start_index = 0, encoded = True, fully_entangled = False, return_matrices = False, discord = False, noise = False, with_permutations = False, format = 'npy'):
        # possible methods:
        # 0: pure separable state
        # 1: W state
        # 2: GHZ state
        # 3: random entanglement
        
        self._initialize_generator(examples, qubits_num, with_permutations)
        params = self.prepare_circuits(random_gates, specified_method, fully_entangled)
        dens_matrices = self.execute_circuits(noise)
        if save_data_dir != None:
            for i in range(len(dens_matrices)):
                m, entangled_qbits = params[i]
                ro = dens_matrices[i]

                save_dens_matrix_with_labels(self.num_qubits, f"dens{start_index + i}", ro, self.methods[m].__name__, entangled_qbits, save_data_dir, separate_bipart=encoded, discord=discord, format=format)
            if return_matrices:
                return dens_matrices
        else:
            return dens_matrices


    def execute_circuits(self, noise):
        if noise:
            executed = execute(self.circuits, self.backend, noise_model = self._noise()).result()
        else:
            executed = execute(self.circuits, self.backend).result()
        state_vectors = [executed.get_statevector(i) for i in range(len(self.circuits))]
        dens_matrices = [DensityMatrix(vec) for vec in state_vectors]
        return dens_matrices


    def prepare_circuits(self, random_gates, specified_method, fully_entangled):
        params = [] 

        for i in range(self.num_examples):
            if specified_method != None and specified_method >= 0 and specified_method < 4:
                m = specified_method
            else:    
                m = random.choice(range(4))
            if m == 0:
                self.methods[m](self.circuits[i], random_gates)
                entangled_qbits = None
            else:
                if fully_entangled:
                    entangled_qbits = self.num_qubits
                    if m == 3:
                         self._generate_random_state(self.circuits[i], entangled_qbits, random_gates, 'full_entanglement')
                    else:
                        self.methods[m](self.circuits[i], entangled_qbits, random_gates)
                else:
                    entangled_qbits = random.choice(range(2, self.num_qubits + 1))
                    self.methods[m](self.circuits[i], entangled_qbits, random_gates)

            params.append((m, entangled_qbits))

            if self.with_permutations:
                qubits_perm = list(permutations(list(np.arange(self.num_qubits))))
                rand_perm = qubits_perm[np.random.randint(len(qubits_perm))]
                permute(rand_perm, self.circuits[i])
        return params


    @generate_with_assertion(1000)
    def generate_random_haar_matrices(self, qubits_num, examples, save_data_dir = None, start_index = 0, encoded = True, return_matrices = False, discord = False, with_permutations = False, format = 'npy'):
        self._initialize_generator(examples, qubits_num, with_permutations)
        dim = 2**self.num_qubits

        states = [random_statevector(dim) for i in range(self.num_examples)]
        matrices = [DensityMatrix(vec) for vec in states]

        if self.with_permutations:
            qubits_perm = list(permutations(list(np.arange(self.num_qubits))))
            for i in range(len(matrices)):
                rand_perm = qubits_perm[np.random.randint(len(qubits_perm))]
                matrices[i] = permute_matrix(rand_perm, matrices[i])

        if save_data_dir == None:
            return matrices
        else:
            for i in range(len(matrices)):
                save_dens_matrix_with_labels(self.num_qubits, f"dens{start_index + self.num_permutations*i}", matrices[i], f"random_vector", "unknown", save_data_dir, separate_bipart=encoded, discord=discord, format=format)

                if return_matrices:
                    return matrices


    def generate_random_separable_orthogonal_pairs(self, qubits_num, examples, with_permutations = False):
        self._initialize_generator(examples, qubits_num, with_permutations)
        assert self.num_examples % 2 == 0, "Number of examples must be even!"

        for i in range(self.num_examples // 2):
            self._generate_random_ps_orthogonal_pair(self.circuits[2*i:2*(i+1)], 1)

            if self.with_permutations:
                qubits_perm = list(permutations(list(np.arange(self.num_qubits))))
                rand_perm = qubits_perm[np.random.randint(len(qubits_perm))]

                permute(rand_perm, self.circuits[i])
                permute(rand_perm, self.circuits[i + 1])
        
        executed = execute(self.circuits, self.backend).result()
        state_vectors1 = [executed.get_statevector(2*i) for i in range(len(self.circuits) // 2)]
        state_vectors2 = [executed.get_statevector(2*i + 1) for i in range(len(self.circuits) // 2)]
        dens_matrices1 = [DensityMatrix(vec) for vec in state_vectors1]
        dens_matrices2 = [DensityMatrix(vec) for vec in state_vectors2]

        return (dens_matrices1, dens_matrices2)
