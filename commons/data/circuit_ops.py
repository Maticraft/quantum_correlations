from functools import reduce
import random
from math import pi

import numpy as np
from scipy.special import comb
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector


def F_gate(circuit, qi, qj, n, k) :
    theta = np.arccos(np.sqrt(1/(n-k+1)))
    circuit.ry(-theta, qj)       
    circuit.cz(qi, qj)
    circuit.ry(theta, qj)
    circuit.barrier(qi)


# |W> = 1/sqrt(3) (|001> + |010> + |100>)
def W_state(qbits, circuit):
    n = len(qbits)
    
    circuit.x(qbits[n - 1])

    for i in range(n, 1, -1):
        F_gate(circuit, qbits[i-1], qbits[i-2], n, n - i + 1)

    for j in range(n, 1, -1):
        circuit.cx(qbits[j-2], qbits[j-1])


# |GHZ> = 1/sqrt(2) (|000> + |111>)
def GHZ_state(qbits, circuit):
    n = len(qbits)

    circuit.h(qbits[0])

    for i in range(0, n-1):
        circuit.cx(qbits[i], qbits[i+1])


# Random control unitary gates 
def random_entanglement(qbits, circuit, max_ctrl_gates, gates_mode = 'random'):
    if gates_mode == 'random':
        if max_ctrl_gates == (-1):
            n = len(qbits)
            max_ctrl_gates = comb(n, 2, exact=True) * 2
        
        num_ctrl_gates = random.choice(list(np.arange(1, max_ctrl_gates)))
    elif gates_mode == 'all':
        num_ctrl_gates = max_ctrl_gates
    else:
        raise ValueError("Wrong gates mode")

    for i in range(num_ctrl_gates):
        if gates_mode == 'random':
            q = random.choice(qbits)
            q2 = random.choice([qbit for qbit in qbits if qbit != q])
        elif gates_mode == 'all':
            q = qbits[i % len(qbits)]
            q2 = qbits[(i+1) % len(qbits)]
        

        theta = random.uniform(0, 2*pi)
        phi = random.uniform(0, 2*pi)
        lamb = random.uniform(0, 2*pi)
        gamma = random.uniform(0, 2*pi)

        circuit.cu(theta, phi, lamb, gamma, q, q2)


# Random entanglement of a given pair of qubits
def random_pair_entanglement(qbits, circuit):
    theta = random.uniform(0, 2*pi)
    phi = random.uniform(0, 2*pi)
    lamb = random.uniform(0, 2*pi)
    gamma = random.uniform(0, 2*pi)

    inverted = bool(random.getrandbits(1))
    if inverted:
        circuit.cu(theta, phi, lamb, gamma, qbits[1], qbits[0])
    else:
        circuit.cu(theta, phi, lamb, gamma, qbits[0], qbits[1])


# Pure, separable, random state
def local_randomization(qbits, circuit, num_gates):
    for q in qbits:
        for i in range(num_gates):
            theta = random.uniform(0, 2*pi)
            phi = random.uniform(0, 2*pi)
            lamb = random.uniform(0, 2*pi)

            circuit.u(theta, phi, lamb, q)


# Local randomization for multiple circuits
def multiple_local_randomization(qbits, circuits, num_gates, preserve_orthogonality = False):
    for q in qbits:
        for i in range(num_gates):
            theta = random.uniform(0, 2*pi)
            lamb = random.uniform(0, 2*pi)
            if preserve_orthogonality:
                phi = 0
            else:
                phi = random.uniform(0, 2*pi)

            for circuit in circuits:
                circuit.u(theta, phi, lamb, q)


# Permutation of a circuit
# given a permutation (qubits list with new order) permutes a circuit to reflect that new order of qubits
# e.g. for 4 qubits circuit permutation [0, 1, 3, 2] results in swapping 3rd and 4th qubits
def permute(permutation, circuit):
    for i in range(circuit.num_qubits):
        if permutation[i] > i:
            circuit.swap(i, permutation[i])


# Permutation of a density matrix
# applies the permute operation for the given density matrix (DensityMatrix from qiskit)
def permute_matrix(permutation, matrix):
    qc = QuantumCircuit(len(permutation))
    permute(permutation, qc)
    return matrix.evolve(qc, list(np.arange(len(permutation))))


# Local random operations on single qubits
def local_randomize_matrix(qbits, matrix, num_gates):
    qc = QuantumCircuit(len(qbits))
    local_randomization(list(np.arange(len(qbits))), qc, num_gates)
    return matrix.evolve(qc, qbits)


def rotate_matrix(qbits, matrix, theta, phi, lam):
    qc = QuantumCircuit(len(qbits))
    qc.u(theta, phi, lam, qbits)
    return matrix.evolve(qc, qbits)


def expand_matrix(matrix, num_qubits):
    simple_state = Statevector(np.array([1, 0])) # state |0>
    simple_state_matrix = DensityMatrix(simple_state)
    extension_matrix = reduce(np.kron, [simple_state_matrix.data for i in range(num_qubits)])
    extension_matrix = DensityMatrix(extension_matrix)
    return matrix.expand(extension_matrix)
