from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np

import torch

from qiskit.quantum_info import DensityMatrix, random_statevector, partial_trace
from qiskit import *
from qiskit import Aer

from commons.data.circuit_ops import permute_matrix, local_randomize_matrix, local_randomization, random_entanglement


def all_perms(rhos, specified_inds = None):
    rhos = torch.squeeze(rhos[:,0,:,:] + rhos[:,1,:,:]*1.j, dim=1).tolist()
    dms =  [DensityMatrix(matrix) for matrix in rhos]
    dim = int(round(np.log2(dms[0].dim)))
    
    permuted_dms = []
    perms = list(permutations(range(dim)))
    if specified_inds != None:
        if type(specified_inds) == int:
            perms = [np.array(perms)[specified_inds]]
        else:
            perms = np.array(perms)[specified_inds]

    for perm in perms:
        permuted_dms.append([permute_matrix(perm, dm).data.tolist() for dm in dms])

    tpermuted_dms = torch.tensor(permuted_dms)
    return torch.stack((tpermuted_dms.real, tpermuted_dms.imag), dim=2)


def loc_op_circ(rhos):
    rhos = torch.squeeze(rhos[:,0,:,:] + rhos[:,1,:,:]*1.j, dim=1).tolist()
    dim = int(round(np.log2(len(rhos[0]))))

    rand_rhos = [local_randomize_matrix(list(np.arange(dim)), DensityMatrix(rho), 2).data.tolist() for rho in rhos]
    trand_rhos = torch.tensor(rand_rhos)
    return torch.stack((trand_rhos.real, trand_rhos.imag), dim= 1)


def loc_op(rhos):
    """
    performs local operations (rotations) that do not change entanglement (from definition)
    :param rhos: [ndarry], input density matrices
    :return: [ndarray], transformed density matrices
    """
    rhos = torch.squeeze(rhos[:,0,:,:] + rhos[:,1,:,:]*1.j, dim=1) 

    assert rhos.shape[1] == rhos.shape[2], "density matrix should be square"
    dim = int(round(np.log2(rhos.shape[1])))
    rotations = np.array([1.])
    thetas = np.random.rand(dim,3)*np.pi*2.
    for theta in thetas:
        c1, s1 = np.cos(theta[0]), np.sin(theta[0])
        c2, s2 = np.cos(theta[1]), np.sin(theta[1])
        c3, s3 = np.cos(theta[2]), np.sin(theta[2])
        r = np.array([[(c2+1.j*s2)*c1, (c3+1.j*s3)*s1], [-(c3-1.j*s3)*s1, (c2-1.j*s2)*c1]])
        rotations = np.kron(rotations, r)
    trotations = torch.tensor(rotations)
    trotationsH = trotations.conj().T
    # I am not sure if detach is needed here...
    rot_rhos = torch.stack([torch.mm(trotationsH, torch.mm(rho.detach(), trotations)) for rho in rhos], dim=0)
    return torch.stack((rot_rhos.real, rot_rhos.imag), dim=1)


#hardcoded aproximate method for 4 and 5 qubits
def extend_label(label, num_qubits):
    label = label.tolist()
    if num_qubits == 4:
        new_label = [[l[0], l[1], l[2], 0., l[2], l[1], l[0]] for l in label]
    elif num_qubits == 5:
        new_label = [[l[0], l[1], l[2], 0., 0., l[2], l[1], l[0], l[0], l[0], l[1], l[1], l[2], l[2], 0.] for l in label]
    return torch.tensor(new_label).double()


def extend_states(rhos, label, new_num_qubits):
    rhos = torch.squeeze(rhos[:,0,:,:] + rhos[:,1,:,:]*1.j).numpy()
    num_qbits = int(round(np.log2(len(rhos[0]))))

    dm = DensityMatrix(random_statevector(2)).data
    for i in range(new_num_qubits - num_qbits - 1):
        dm_i = DensityMatrix(random_statevector(2)).data
        dm = np.kron(dm, dm_i)

    ext_rhos = np.array([np.kron(rho, dm) for rho in rhos])

    """
    # Exact target
    neg_values = np.array([global_entanglement_bipartitions(DensityMatrix(rho), 'negativity', return_separate_outputs=True)[1] for rho in ext_rhos]) 
    ext_target = torch.tensor(neg_values > 0.0001).double()
    """

    # Approximate target
    ext_target = extend_label(label, new_num_qubits)

    text_rhos = torch.tensor(ext_rhos)
    return torch.stack((text_rhos.real, text_rhos.imag), dim= 1), ext_target


# pure state (circuit) represantation of the mixed state
def pure_rep(label, num_qubits = 3):
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits, 'c')
    circuits = [QuantumCircuit(qr, cr) for ex in range(len(label))]
    backend = Aer.get_backend('statevector_simulator')

    for i in range(len(label)):
        qbits = list(np.arange(num_qubits))
        ent_qbits = list(np.where(label[i].numpy() == 1)[0][:num_qubits])

        local_randomization(qr[qbits], circuits[i], 1)
        if len(ent_qbits) >=2:
            idx = i
            random_entanglement(qr[ent_qbits], circuits[i], len(ent_qbits) - 1, 'all')
        local_randomization(qr[qbits], circuits[i], 2)

    executed = execute(circuits, backend).result()
    state_vectors = [executed.get_statevector(i) for i in range(len(circuits))]
    dens_matrix = [DensityMatrix(state_vector) for state_vector in state_vectors]

    torch_dm = torch.tensor([dm.data.tolist() for dm in dens_matrix])

    return torch.stack((torch_dm.real, torch_dm.imag), dim = 1) 


def amplitude_reflection(rho):
    """
    Apply amplitude reflection for the single qubit density matrix rho
    """
    rho_tmp = rho.copy()
    rho_tmp[0,0] = rho[1,1]
    rho_tmp[1,1] = rho[0,0]
    return rho_tmp


def amplitude_symmetry(rhos):
    rhos = torch.squeeze(rhos[:,0,:,:] + rhos[:,1,:,:]*1.j, dim=1).tolist() 
    num_qbits = int(round(np.log2(len(rhos[0]))))

    k = np.random.randint(1, num_qbits + 1)
    qubits_for_sym = np.random.choice(num_qbits, k, replace=False)

    rhos_sym = [partial_trace(DensityMatrix(rho), [x for x in range(num_qbits) if x != 0]).data for rho in rhos]
    
    if 0 in qubits_for_sym:
        rhos_sym = [amplitude_reflection(rho_sym) for rho_sym in rhos_sym]

    for q in range(1, num_qbits):
        rhos_q = [partial_trace(DensityMatrix(rho), [x for x in range(num_qbits) if x != q]).data for rho in rhos]

        if q in qubits_for_sym:
            rhos_q = [amplitude_reflection(rho_q) for rho_q in rhos_q]
        
        rhos_sym = [np.kron(rho_q, rho_sym) for rho_sym, rho_q in zip(rhos_sym, rhos_q)]
    
    trhos_sym = torch.tensor(np.stack(rhos_sym, axis=0))
    return torch.stack((trhos_sym.real, trhos_sym.imag), dim= 1)
    

def amplitude_reflection_dm(rho, qubit):
    """
    Apply amplitude reflection for given qubit for the density matrix rho
    """
    rho_sym = rho.clone().detach()
    d = 2**(qubit+1)
    i = 0
    while i < len(rho):
        j = 0
        while j < len(rho[i]):
            rho_tmp = rho[i:i+d, j:j+d].clone().detach()

            tmp1 = rho_tmp[:d//2, :d//2].clone().detach().numpy()
            amp1, phase1 = np.abs(tmp1), np.angle(tmp1)

            tmp2 = rho_tmp[d//2:, d//2:].clone().detach().numpy()
            amp2, phase2 = np.abs(tmp2), np.angle(tmp2)

            rho_tmp[:d//2, :d//2] = torch.tensor(amp2*np.exp(1.j*phase1))
            rho_tmp[d//2:, d//2:] = torch.tensor(amp1*np.exp(1.j*phase2))

            rho_sym[i:i+d, j:j+d] = rho_tmp
            j += d
        i += d
    return rho_sym


def amplitude_symmetry_dm(rhos, qubits_for_sym):
    rhos_sym = torch.squeeze(rhos[:,0,:,:] + rhos[:,1,:,:]*1.j, dim=1)

    for q in qubits_for_sym:
        rhos_sym = [amplitude_reflection_dm(rho_sym, q) for rho_sym in rhos_sym]
    
    trhos_sym = torch.stack(rhos_sym, dim=0)
    return torch.stack((trhos_sym.real, trhos_sym.imag), dim= 1)


def torch_fidelity(rho1, rho2):
    unitary1, singular_values, unitary2 = torch.linalg.svd(rho1)
    diag_func_singular = torch.diag(torch.sqrt(singular_values)).to(torch.cdouble)
    s1sqrt =  unitary1.matmul(diag_func_singular).matmul(unitary2)   

    unitary1, singular_values, unitary2 = torch.linalg.svd(rho2)
    diag_func_singular = torch.diag(torch.sqrt(singular_values)).to(torch.cdouble)
    s2sqrt =  unitary1.matmul(diag_func_singular).matmul(unitary2)   

    fid = torch.linalg.norm(s1sqrt.matmul(s2sqrt), ord="nuc") ** 2
    return fid.to(torch.double)


def plot_loss(train_loss, validation_loss, title, log_scale = False):
    plt.grid(True)
    plt.xlabel("subsequent epochs")
    plt.ylabel('average loss')
    plt.plot(range(1, len(train_loss)+1), train_loss, 'o-', label='training')
    plt.plot(range(1, len(validation_loss)+1), validation_loss, 'o-', label='validation')
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.title(title)
    plt.show()


def save_acc(file_path, x, accuracies, write_mode = "a"):
    acc_str = ""
    for acc in accuracies:
        acc_str += "  " + str(acc)

    with open(file_path, write_mode) as f:
        f.write(str(x) + acc_str + "\n")


def load_acc(file_path, skiprows=0):
    with open(file_path, "r") as f:
        lines = f.readlines()
    lines = lines[skiprows:]
    lines_sep = [[float(x) for x in line.split("  ")] for line in lines]
    return np.array(lines_sep)
