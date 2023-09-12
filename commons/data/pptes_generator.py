from itertools import permutations

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector

from commons.data.savers import save_dens_matrix_with_labels
from commons.data.decorators import generate_with_assertion
from commons.data.circuit_ops import local_randomize_matrix, permute_matrix
from commons.data.pure_states_generator import PureStatesGenerator


class PPTESGenerator():
    def _initialize_generator(self, examples, qubits_num, with_permutations):
        self.num_examples = examples
        self.num_qubits = qubits_num
        self.with_permutations = with_permutations

    # those are in fact mixed states (PPTES)
    @generate_with_assertion()
    def generate_3qbits_pptes(self, qubits_num, examples, save_data_dir, entanglement_class = "Horodecki", randomization = False, start_index = 0, encoded = True, ppt = True, discord = False, with_permutations = False, format = 'npy'):
        self._initialize_generator(examples, qubits_num, with_permutations)
        if self.num_qubits != 3:
            raise ValueError("Bound entangled state implemented only for 3 qubits")
        
        dim = 2**self.num_qubits

        if save_data_dir == None:
            matrices = []

        for i in range(self.num_examples):

            # Horodecki 2 x 4 pptes
            if entanglement_class == 'Horodecki':
                dm = self._generate_single_Horodecki_state(dim)

            elif entanglement_class == "Acin":
                dm = self._generate_single_Acin_state(dim)

            elif entanglement_class == "BennetBE":
                dm = self._generate_single_Bennet_bound_entangled_state()

            else:
                raise ValueError("Entanglement class must be either Horodecki or Acin or BennetBE")

            if randomization:
                dm = local_randomize_matrix(list(np.arange(self.num_qubits)), dm, 2)
                dm = DensityMatrix(dm.data / dm.trace())

            if self.with_permutations:
                qubits_perm = list(permutations(list(np.arange(self.num_qubits))))
                rand_perm = qubits_perm[np.random.randint(len(qubits_perm))]
                dm = permute_matrix(rand_perm, dm)

            if save_data_dir != None:
                save_dens_matrix_with_labels(self.num_qubits, f"dens{start_index + i}", dm, f"{entanglement_class}_ppt_entangled_state", "unknown", save_data_dir, ppt=ppt, separate_bipart=encoded, discord=discord, format=format)
            else:
                matrices.append(dm)
        
        if save_data_dir == None:
            return matrices 
        

    @staticmethod
    def _generate_single_Horodecki_state(dim):
        b = np.random.uniform(0.1, 0.9)
        rho = np.zeros((dim, dim))

        rho[0, 0] = b
        rho[0, 5] = b
        rho[1, 1] = b
        rho[1, 6] = b
        rho[2, 2] = b
        rho[2, 7] = b
        rho[3, 3] = b      
        rho[4, 4] = (1 + b)/2
        rho[4, 7] = np.sqrt(1 - b*b)/2
        rho[5, 0] = b
        rho[5, 5] = b
        rho[6, 1] = b
        rho[6, 6] = b
        rho[7, 2] = b
        rho[7, 4] = np.sqrt(1 - b*b)/2   
        rho[7, 7] = (1 + b)/2

        dm = DensityMatrix(rho/(7*b + 1))
        return dm
    

    @staticmethod
    def _generate_single_Acin_state(dim):
        a = np.random.uniform(0.01, 100.)
        b = np.random.uniform(0.01, 100.)
        c = np.random.uniform(0.01, 100.)
        n = 2 + a + 1/a + b + 1/b + c + 1/c
        rho = np.zeros((dim, dim))

        rho[0, 0] = 1
        rho[0, 7] = 1
        rho[1, 1] = a 
        rho[2, 2] = b
        rho[3, 3] = c
        rho[4, 4] = 1/c
        rho[5, 5] = 1/b
        rho[6, 6] = 1/a
        rho[7, 0] = 1
        rho[7, 7] = 1

        dm = DensityMatrix(rho/n)
        return dm


    @staticmethod
    def _generate_single_Bennet_bound_entangled_state():
        v1 = Statevector([1, 0, 0, 0, 0, 0, 0, 0])
        v2 = Statevector([0, 0, 0, 0, -0.5, -0.5, 0.5, 0.5])
        v3 = Statevector([0, 0, -0.5, 0.5, 0, 0, -0.5, 0.5])
        v4 = Statevector([0, -0.5, 0, -0.5, 0, 0.5, 0, 0.5])

        rho1 = DensityMatrix(v1).data
        rho2 = DensityMatrix(v2).data
        rho3 = DensityMatrix(v3).data
        rho4 = DensityMatrix(v4).data

        rho = np.eye(8, 8) - np.sum(np.stack([rho1, rho2, rho3, rho4], axis=0), axis=0)
        dm = DensityMatrix(rho)
        return dm    


    @generate_with_assertion()
    def generate_extended_3q_pptes(self, qubits_num, examples, save_data_dir, entanglement_class = "Horodecki", randomization = False, start_index = 0, encoded = True, ppt = True, discord = False, with_permutations = False, format = 'npy'):
        assert qubits_num > 3, "Number of qubits must be greater than 3"
        dms_3q = self.generate_3qbits_pptes(3, examples, None, entanglement_class, randomization, start_index, encoded, ppt)
        self._initialize_generator(examples, qubits_num, with_permutations)

        gen = PureStatesGenerator()
        dms_1q = gen.generate_circuit_matrices(1, self.num_examples*(self.num_qubits - 3), save_data_dir=None, specified_method=0)

        not_ent_qbits = list(np.arange(3, self.num_qubits))

        for i, dm in enumerate(dms_3q):
            ext_dm = dm.data
            for j in range(self.num_qubits - 3):
                ext_dm = np.kron(ext_dm, dms_1q[j*self.num_examples + i])

            ext_dm = DensityMatrix(ext_dm)


            if self.with_permutations:
                qubits_perm = list(permutations(list(np.arange(self.num_qubits))))
                rand_perm = qubits_perm[np.random.randint(len(qubits_perm))]
                not_ent_qbits = [rand_perm.index(x) for x in not_ent_qbits]
                ext_dm = permute_matrix(rand_perm, ext_dm)

            save_dens_matrix_with_labels(self.num_qubits, f"dens{start_index + i}", ext_dm, f"extended_{entanglement_class}_pptes", f"3/{self.num_qubits}", save_data_dir, ppt = ppt, separate_bipart=encoded, not_ent_qbits=not_ent_qbits, discord = discord, format=format)


    # Generalization of Horodecki 2 x 4 PPTES for 2 x d dimension, i.e. this method applicates for arbitrary number of qubits
    @generate_with_assertion()
    def generate_2xd_pptes(self, qubits_num, examples, save_data_dir, randomization = False, start_index = 0, encoded = True, discord = False, with_permutations = False, format = 'npy'):
        self._initialize_generator(examples, qubits_num, with_permutations)
        dim = 2**self.num_qubits
        d = dim // 2

        for ex in range(self.num_examples):

            dm = self._generate_single_2xd_pptes(dim, d)

            if randomization:
                dm = local_randomize_matrix(list(np.arange(self.num_qubits)), dm, 2)

            if self.with_permutations:
                qubits_perm = list(permutations(list(np.arange(self.num_qubits))))
                rand_perm = qubits_perm[np.random.randint(len(qubits_perm))]
                dm = permute_matrix(rand_perm, dm)

            save_dens_matrix_with_labels(self.num_qubits, f"dens{start_index + ex}", dm, f"2xd_pptes", "unknown", save_data_dir, ppt=True, separate_bipart=encoded, discord = discord, format=format)


    @staticmethod
    def _generate_single_2xd_pptes(dim, d):
        b = np.random.uniform(0.1, 0.9)
        rho = np.zeros((dim, dim))

        for i in range(0, d-1):
            rho[i, i] += b
            rho[i, d + i + 1] += b
            rho[d + i + 1, i] += b
            rho[d + i + 1, d + i + 1] += b
            
        rho[d, d] += b
        rho[0, 0] += (1 - b)/2
        rho[0, d - 1] += np.sqrt(1 - b*b)/2
        rho[d - 1, 0] += np.sqrt(1 - b*b)/2
        rho[d - 1, d - 1] += (1 + b)/2

        rho = 1/((2*d - 1)*b + 1) * rho
        dm = DensityMatrix(rho)
        return dm
