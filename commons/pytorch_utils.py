import os
from itertools import permutations
from math import factorial, log2

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from qiskit.quantum_info import DensityMatrix, random_statevector, partial_trace
from qiskit import *
from qiskit import Aer

from commons.data.circuit_ops import permute_matrix, local_randomize_matrix, local_randomization, random_entanglement


class DensityMatricesDataset(Dataset):

    def __init__(self, dictionary, root_dir, metrics, threshold, data_limit = None):
        self.dictionary = self.load_dict(dictionary)
        self.root_dir = root_dir
        self.metrics = metrics
        self.threshold = threshold
        self.data_limit = data_limit

        if self.metrics == "global_entanglement":
            self.label_pos = 3

        elif self.metrics == "von_Neumann":
            self.label_pos = 4

        elif self.metrics == "concurrence":
            self.label_pos = 5

        elif self.metrics == "negativity":
            self.label_pos = 6
        
        elif self.metrics == "realignment":
            self.label_pos = 7
            
        elif self.metrics == "discord":
            self.label_pos = 8

        elif self.metrics == "trace":
            self.label_pos = 9

        else:
            raise ValueError('Wrong metrics')
      
    def __len__(self):
        if self.data_limit != None:
            return self.data_limit
        else:
            return len(self.dictionary)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        matrix_name = os.path.join(self.root_dir, self.dictionary[idx][0] + ".npy")
        matrix = np.load(matrix_name)
        matrix_r = np.real(matrix)
        matrix_im = np.imag(matrix)

        tensor = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0))

        label = float(self.dictionary[idx][self.label_pos])
        if label > self.threshold:
            label = 1
        else:
            label = 0
        label = torch.tensor(label).double()
        label = label.unsqueeze(0)

        return (tensor, label)


    def load_dict(self, filepath):
        with open(filepath, 'r') as dictionary:
            data = dictionary.readlines()

        parsed_data = [row.rstrip("\n").split(', ') for row in data]

        return parsed_data


class BipartitionMatricesDataset(Dataset):

    def __init__(self, dictionary, root_dir, threshold, data_limit = None):
        self.dictionary = self.load_dict(dictionary)[:data_limit]
        self.root_dir = root_dir
        self.data_limit = data_limit
        self.bipart_num = len(self.dictionary[0]) - 1
        self.threshold = threshold


    def __len__(self):
        if self.data_limit != None:
            return self.data_limit
        else:
            return len(self.dictionary)


    def __getitem__(self, idx):
      if torch.is_tensor(idx):
        idx = idx.tolist()

      filename = self.dictionary[idx][0] + ".npy"

      matrix_name = os.path.join(self.root_dir, filename)
      matrix = np.load(matrix_name)
      matrix_r = np.real(matrix)
      matrix_im = np.imag(matrix)

      tensor = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0))

      label = torch.tensor([1. if float(self.dictionary[idx][i]) > self.threshold else 0. for i in range(1, len(self.dictionary[0]))]).double()
    
      return (tensor, label)


    def load_dict(self, filepath):
      
      with open(filepath, 'r') as dictionary:
        data = dictionary.readlines()

      parsed_data = [row.rstrip("\n").split(', ') for row in data]

      return parsed_data


def rho_reconstruction(x, separator_output):
    ch = separator_output[0].size()[1] // 2
    rho = torch.zeros_like(x[:,0,:,:], dtype = torch.cdouble)

    for i in range(ch):
        dms = separator_output[0]
        rho_real = dms[:, i, :, :]
        rho_imag = dms[:, ch + i, :, :]
        rho_i = torch.complex(rho_real, rho_imag)
            
        for j in range(1, len(separator_output)):
            dms = separator_output[j]
            ch = dms.size()[1] // 2
            rho_real = dms[:, i, :, :]
            rho_imag = dms[:, ch + i, :, :]
            rho_j = torch.complex(rho_real, rho_imag)      
            rho_i = torch.stack([torch.kron(rho_i[k], rho_j[k]) for k in range(len(rho_i))])
        rho += rho_i

    rho = rho / ch
    rho = torch.stack((rho.real, rho.imag), dim = 1)
    return rho


def sep_met(output, data):
    ch = output[0].size()[1] // 2
    rho = torch.zeros_like(data[:,0,:,:], dtype = torch.cdouble)
    trace_loss = 0
    hermitian_loss = 0

    for i in range(ch):
        dms = output[0]
        rho_real = dms[:, i, :, :]
        rho_imag = dms[:, ch + i, :, :]
        rho_i = torch.complex(rho_real, rho_imag)
            
        for j in range(1, len(output)):
            dms = output[j]
            ch = dms.size()[1] // 2
            rho_real = dms[:, i, :, :]
            rho_imag = dms[:, ch + i, :, :]
            rho_j = torch.complex(rho_real, rho_imag)
            trace_loss += torch.mean(torch.stack([torch.abs(torch.trace(rho_j[k]) - 1.) for k in range(len(rho_j))]))  
            hermitian_loss += torch.mean(torch.abs(torch.conj(torch.transpose(rho_j, -1, -2)) - rho_j))          
            rho_i = torch.stack([torch.kron(rho_i[k], rho_j[k]) for k in range(len(rho_i))])
        rho += rho_i

    rho = rho / ch
    rho = torch.stack((rho.real, rho.imag), dim = 1)
    rho_diff = torch.abs(rho - data)
    metric = torch.mean(rho_diff.view(data.size()[0], -1), dim=1, keepdim=True) + 0.1 * (trace_loss + hermitian_loss) / (ch * (len(output) - 1))
    return metric, rho_diff

    

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
    num_qbits = int(round(np.log2(len(rhos_sym[0]))))

    for q in qubits_for_sym:
        rhos_sym = [amplitude_reflection_dm(rho_sym, q) for rho_sym in rhos_sym]
    
    trhos_sym = torch.stack(rhos_sym, dim=0)
    return torch.stack((trhos_sym.real, trhos_sym.imag), dim= 1)


def trace_reconstruction(rhos):
    rhos = torch.squeeze(rhos[:,0,:,:] + rhos[:,1,:,:]*1.j, dim=1).tolist() 
    num_qbits = int(round(np.log2(len(rhos[0]))))

    rhos_sym = [partial_trace(DensityMatrix(rho), [x for x in range(num_qbits) if x != 0]).data for rho in rhos]

    for q in range(1, num_qbits):
        rhos_q = [partial_trace(DensityMatrix(rho), [x for x in range(num_qbits) if x != q]).data for rho in rhos]        
        rhos_sym = [np.kron(rho_q, rho_sym) for rho_sym, rho_q in zip(rhos_sym, rhos_q)]
    
    trhos_sym = torch.tensor(np.stack(rhos_sym, axis=0))
    return torch.stack((trhos_sym.real, trhos_sym.imag), dim= 1)


def trace_predict(data, threshold, criterion = 'L1', return_measure_value = False):
  
    data_reconstructed = trace_reconstruction(data)

    if criterion == 'bures':
        data_complex = torch.complex(data[:, 0, :, :], data[:, 1, :, :])
        rho = torch.complex(data_reconstructed[:,0,:,:], data_reconstructed[:,1,:,:])

        loss = torch.stack([2*(torch.abs(1 - torch.sqrt(torch_fidelity(rho[i], data_complex[i])))) for i in range(data_complex.size()[0])])
    else:
        criterion = nn.L1Loss(reduction='none')
        loss = torch.mean(criterion(data_reconstructed, data), dim=(1,2,3))

    prediction = torch.ones(data.size()[0])

    for ex in range(data.size()[0]):
        if loss[ex] < threshold:
            prediction[ex] = 0

    if return_measure_value:
        return prediction, loss
    else:
        return prediction


def test_trace_predictions(test_loader, criterion, threshold, message, confusion_matrix = False):
    correct = 0
    test_loss = 0.

    if confusion_matrix:
        conf_matrix = np.zeros((2, 2))

    for data, target in test_loader:
        prediction, loss = trace_predict(data, threshold, criterion, True)
        test_loss += torch.mean(loss)
        prediction = prediction.unsqueeze(1)
        correct += prediction.eq(target).sum().item()

        if confusion_matrix:
            for i, j in zip(target, prediction):
                conf_matrix[int(i), int(j)] += 1

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)

    print('{}: Average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(message, test_loss, correct, len(test_loader.dataset), acc))
    if confusion_matrix:
        print('Confusion matrix:\n{}'.format(conf_matrix))
        return test_loss, acc, conf_matrix

    return test_loss, acc


def torch_fidelity(rho1, rho2):
    unitary1, singular_values, unitary2 = torch.linalg.svd(rho1)
    diag_func_singular = torch.diag(torch.sqrt(singular_values)).to(torch.cdouble)
    s1sqrt =  unitary1.matmul(diag_func_singular).matmul(unitary2)   

    unitary1, singular_values, unitary2 = torch.linalg.svd(rho2)
    diag_func_singular = torch.diag(torch.sqrt(singular_values)).to(torch.cdouble)
    s2sqrt =  unitary1.matmul(diag_func_singular).matmul(unitary2)   

    fid = torch.linalg.norm(s1sqrt.matmul(s2sqrt), ord="nuc") ** 2
    return fid.to(torch.double)



def train_vector_siamese(model, device, train_loader, optimizer, criterion, epoch_number, interval, loc_op_flag = False, reduced_perms_num = None, pure_representation = False, biparts = 'separate'):
    model.train()
    train_loss = 0.

    
    if epoch_number <= 10:
        lambda_1 = 0
    else:
        lambda_1 = epoch_number/20
    

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()

        if reduced_perms_num != None:
            if biparts == 'single':
                inds = [0] + list(np.random.choice(np.arange(1, factorial(int(round(np.log2(model.dim))) - 1)), size = reduced_perms_num, replace=False))
            else:
                inds = [0] + list(np.random.choice(np.arange(1, len(model.perms)), size = reduced_perms_num, replace=False))
            perm_data = all_perms(data, inds).double().to(device)
        else:
            perm_data = all_perms(data).double().to(device)  

        
        if pure_representation and epoch_number > 10:
            pcm = pure_rep(target, int(log2(model.dim))).double().to(device)
            perm_data = torch.cat((perm_data, torch.torch.unsqueeze(pcm, dim=0)), dim = 0)
        

        if loc_op_flag:
            loc_op_data = loc_op_circ(data).double().to(device)
            outputs = model(torch.cat((perm_data, torch.unsqueeze(loc_op_data, dim=0)), dim=0))
        else:
            outputs = model(perm_data)

        target = target.to(device)

        losses_perm = []

        # separate outputs for different bipartitions
        if biparts == 'separate':
            loss_std = criterion(outputs[0], target)

            for (i1, j1), (i2, j2) in model.matching_indices:
                if reduced_perms_num != None:
                    if j1 in inds and j2 in inds:
                        losses_perm.append(torch.mean(torch.abs(outputs[inds.index(j1),:,i1] - outputs[inds.index(j2),:,i2])))
                else:
                    losses_perm.append(torch.mean(torch.abs(outputs[j1,:,i1] - outputs[j2,:,i2])))

        elif biparts == 'averaged': # single output averaged bipartitions
            loss_std = criterion(outputs[0], target)

            for j1 in range(len(outputs)):
                for j2 in range((len(outputs))):
                    if reduced_perms_num != None:
                        if j1 in inds and j2 in inds:
                            losses_perm.append(torch.mean(torch.abs(outputs[inds.index(j1)] - outputs[inds.index(j2)])))
                    else:
                        losses_perm.append(torch.mean(torch.abs(outputs[j1] - outputs[j2])))

        elif biparts == 'single':
            loss_std = criterion(outputs[0], torch.unsqueeze(target[:,0], dim=1))
            for (i1, j1), (i2, j2) in model.matching_indices:
                if i1 == i2 and i1 == 0:
                    if reduced_perms_num != None:
                        if j1 in inds and j2 in inds:
                            losses_perm.append(torch.mean(torch.abs(outputs[inds.index(j1),:] - outputs[inds.index(j2),:])))
                    else:
                        losses_perm.append(torch.mean(torch.abs(outputs[j1,:] - outputs[j2,:])))
        
        else:
            raise ValueError("Wrong biparts mode")
                    

        if loc_op_flag:
            loss_loc_op1 = torch.mean(torch.abs(outputs[0] - outputs[outputs.size()[0] - 1]))

            if pure_representation and epoch_number > 10:
                loss_pr = torch.mean(torch.abs(outputs[0] - outputs[outputs.size()[0] - 2]))
                loss = loss_std + 0.5*torch.mean(torch.stack(losses_perm)) + 0.5*loss_loc_op1 + lambda_1*loss_pr

            else:
                loss = loss_std + 0.5*torch.mean(torch.stack(losses_perm)) + 0.5*loss_loc_op1 #+ 0.5*loss_loc_op2
        else:
            if pure_representation and epoch_number > 10:
                loss_pr = torch.mean(torch.abs(outputs[0] - outputs[outputs.size()[0] - 1]))
                loss = loss_std + 0.5*torch.mean(torch.stack(losses_perm)) + lambda_1*loss_pr
            else:
                loss = loss_std + 0.5*torch.mean(torch.stack(losses_perm))

        loss.backward()
        optimizer.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss



def train_vector_siamese_with_purificator(model_siam, model_pure, device, train_loader, optimizer_siam, optimizer_pure, epoch_number, interval, loc_op_flag = False, reduced_perms_num = None):
    print("Epoch: {} - Purificator training".format(epoch_number))
    pure_loss = train_purificator(model_pure, device, train_loader, optimizer_pure, nn.L1Loss(), epoch_number, interval)

    print("Epoch: {} - Siamese training".format(epoch_number))
    model_siam.train()
    train_loss = pure_loss

    for batch_idx, (data, target) in enumerate(train_loader):

        target = target.to(device)

        if reduced_perms_num != None:
            inds = [0] + list(np.random.choice(np.arange(1, len(model_siam.perms)), size = reduced_perms_num, replace=False))
            perm_data = all_perms(data, inds).double().to(device)
        else:
            perm_data = all_perms(data).double().to(device)  

        optimizer_siam.zero_grad()
        if loc_op_flag:
            loc_op_data = loc_op_circ(data).double().to(device)
            perm_data = torch.cat((perm_data, torch.unsqueeze(loc_op_data, dim=0)), dim=0)

        purified_data = torch.tensor([model_pure(x) for x in perm_data])
        outputs = model_siam(purified_data)

        criterion = nn.BCELoss()
        loss_std = criterion(outputs[0], target)
        losses_perm = []

        for (i1,j1), (i2, j2) in model_siam.matching_indices:
            if reduced_perms_num != None:
                if j1 in inds and j2 in inds:
                    losses_perm.append(torch.mean(torch.abs(outputs[inds.index(j1),:,i1] - outputs[inds.index(j2),:,i2])))
            else:
                losses_perm.append(torch.mean(torch.abs(outputs[j1,:,i1] - outputs[j2,:,i2])))

        if loc_op_flag:
            loss_loc_op1 = torch.mean(torch.abs(outputs[0] - outputs[outputs.size()[0] - 1]))
            loss = loss_std + 0.5*torch.mean(torch.stack(losses_perm)) + 0.5*loss_loc_op1 
        else:
            loss = loss_std + 0.5*torch.mean(torch.stack(losses_perm))

        loss.backward()
        optimizer_siam.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def train(model, device, train_loader, optimizer, criterion, epoch_number, interval):
    model.train()
    model.to(device)
    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def train_double(model1, model2, device, train_loader, optimizer1, optimizer2, criterion, epoch_number, interval, balance_losses = False, thresh = 0.1):
    model1.train()
    model2.train()

    train_loss1 = 0.
    train_loss2 = 0.


    for batch_idx, (data, target1, target2) in enumerate(train_loader):

        data, target1, target2 = data.to(device), target1.to(device), target2.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        output1 = model1(data)
        loss1 = criterion(output1, target1)

        output2 = model2(data)
        loss2 = criterion(output2, target2)

        if balance_losses and (loss1.item() + thresh) < loss2.item():
            loss2.backward()
            optimizer2.step()
        elif balance_losses and (loss2.item() + thresh) < loss1.item():
            loss1.backward()
            optimizer1.step()
        else:
            loss1.backward()
            optimizer1.step()
            loss2.backward()
            optimizer2.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss1.item(), loss2.item()))

        train_loss1 += loss1.item()
        train_loss2 += loss2.item()


    train_loss1 /= len(train_loader)
    train_loss2 /= len(train_loader)


    print('\nTrain set:\n Average loss1: {:.4f}\nAverage loss2: {:.4f}'.format(train_loss1, train_loss2))
    return train_loss1, train_loss2


def train_siamese(model, device, train_loader, optimizer, criterion, epoch_number, interval):
    model.train()
    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data1, target = data.to(device), target.to(device)
        data2 = loc_op(data).to(device)

        optimizer.zero_grad()
        output1, output2 = model(data1, data2)
        loss1 = criterion(output1, target)
        loss2 = torch.mean(torch.abs(output1 - output2))
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def get_separator_loss(data, output, criterion):
    ch = output[0].size()[1] // 2
    rho = torch.zeros_like(data[:, 0, :, :], dtype = torch.cdouble)

    for i in range(ch):
        dms = output[0]
        rho_real = dms[:, i, :, :]
        rho_imag = dms[:, ch + i, :, :]
        rho_i = torch.complex(rho_real, rho_imag)
    
        for j in range(1, len(output)):
            dms = output[j]
            ch = dms.size()[1] // 2
            rho_real = dms[:, i, :, :]
            rho_imag = dms[:, ch + i, :, :]
            rho_j = torch.complex(rho_real, rho_imag)           
            rho_i = torch.stack([torch.kron(rho_i[k], rho_j[k]) for k in range(len(rho_i))])
        rho += rho_i

    rho = rho / ch
    if criterion == 'bures':
        data_complex = torch.complex(data[:, 0, :, :], data[:, 1, :, :])
        loss = torch.stack([2*(torch.abs(1 - torch.sqrt(torch_fidelity(rho[i], data_complex[i])))) for i in range(data_complex.size()[0])])
    else:
        rho = torch.stack((rho.real, rho.imag), dim = 1)
        loss = criterion(rho, data[:, :2, :, :])

    return loss


def train_separator(model, device, train_loader, optimizer, criterion, epoch_number, interval, use_noise = False, enforce_symmetry = False, train_on_entangled = False):
    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if use_noise:
            output = [model(data, noise = True)]
        outputs = [model(data)]

        if enforce_symmetry:
            k = np.random.randint(1, model.qbits_num + 1)
            qubits_for_sym = np.random.choice(model.qbits_num, k, replace=False)
            data_sym = amplitude_symmetry_dm(data.cpu(), qubits_for_sym).to(device)
            output_sym = model(data_sym)
            outputs.append(output_sym)

        losses = []
        for output_num, output in enumerate(outputs):
            if output_num == 0:
                loss = get_separator_loss(data, output, criterion)
            else:
                loss = get_separator_loss(data_sym, output, criterion)

            losses.append(loss)

        if train_on_entangled:
            std_loss = torch.mean(losses[0])
        else:
            std_loss = torch.mean(losses[0][torch.squeeze(target == 0.)])

        if enforce_symmetry:
            sym_loss = torch.mean(torch.stack([torch.abs(losses[0] - losses[i]) for i in range(1, len(losses))]))
            total_loss = std_loss + 0.5*sym_loss
        else:
            total_loss = std_loss

        total_loss.backward()
        optimizer.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss.item()))

        train_loss += total_loss.item()

    train_loss /= len(train_loader)

    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def get_symmetry_loss(output_1, output_2, qubits_for_sym):
    loss = 0.
    ch = output_1[0].size()[1] // 2
    num_qubits = len(output_1)
    qubits_for_sym_rev = num_qubits - np.array(qubits_for_sym) - 1

    for i in range(num_qubits):
        for j in range(ch):
            rhos_1 = output_1[i]
            rho_real_1 = rhos_1[:, j, :, :]
            rho_imag_1 = rhos_1[:, ch + j, :, :]
            rho_1 = torch.complex(rho_real_1, rho_imag_1)

            rhos_2 = output_2[i]
            rho_real_2 = rhos_2[:, j, :, :]
            rho_imag_2 = rhos_2[:, ch + j, :, :]
            rho_2 = torch.complex(rho_real_2, rho_imag_2)

            rho_1_sym = rho_1

            if i in qubits_for_sym_rev:
                tmp = rho_1[:, 0, 0]
                rho_1_sym[:, 0, 0] = rho_1[:, 1, 1]
                rho_1_sym[:, 1, 1] = tmp

            loss += torch.mean(torch.abs(rho_1_sym - rho_2))
   
    return loss / (ch * len(qubits_for_sym))


def train_siamese_separator(model, device, train_loader, optimizer, criterion, epoch_number, interval):
    model.train()
    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        k = np.random.randint(1, model.qbits_num + 1)
        qubits_for_sym = np.random.choice(model.qbits_num, k, replace=False)

        data_sym = amplitude_symmetry_dm(data.cpu(), qubits_for_sym).to(device)
        output, output_sym = model(data, data_sym)
       
        loss_std = torch.mean(get_separator_loss(data, output, criterion)[torch.squeeze(target == 0.)])
        
        if epoch_number > 2:
            loss_sym = get_symmetry_loss(output, output_sym, qubits_for_sym)
            total_loss = loss_std + .5*loss_sym
        else:
            total_loss = loss_std

        total_loss.backward()
        optimizer.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss.item()))

        train_loss += total_loss.item()

    train_loss /= len(train_loader)

    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


# Model can be trained only on 1|N-1 biseparable states or fully seperable states. In order to train on bispearable states,
# please provide indx of the specific bipartition, which is separable.
def train_separator_bipart(model, device, train_loader, optimizer, criterion, epoch_number, interval, specific_bipartition = None):
    model.train()
    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        q_fac = factorial(model.qbits_num - 1)
        if specific_bipartition == None:
            inds = [q * q_fac for q in range(model.qbits_num)]
        else:
            inds = [q_fac*specific_bipartition]
        biparts_num = len(inds)
        perm_data = all_perms(data, inds).double().to(device)

        diff_loss = torch.zeros(biparts_num).to(device)

        for k in range(biparts_num):
            output = model(perm_data[k])
            ch = output[0].size()[1] // 2

            rho_sep1 = output[0]
            rho_sep2 = output[1]

            rho = torch.zeros_like(data[:, 0, :, :], dtype = torch.cdouble)
        
            for i in range(ch):
                rho1_real = rho_sep1[:, i, :, :]
                rho1_imag = rho_sep1[:, ch + i, :, :]
                rho1_i = torch.complex(rho1_real, rho1_imag)

                rho2_real = rho_sep2[:, i, :, :]
                rho2_imag = rho_sep2[:, ch + i, :, :]
                rho2_i = torch.complex(rho2_real, rho2_imag)

                rho_i = torch.stack([torch.kron(rho1_i[ex], rho2_i[ex]) for ex in range(len(rho1_i))])
                rho += rho_i

            rho = rho / ch
            rho = torch.stack((rho.real, rho.imag), dim = 1)
            diff_loss[k] = criterion(rho, perm_data[k])

        loss = torch.mean(diff_loss)
        loss.backward()
        optimizer.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def train_purificator(model, device, train_loader, optimizer, criterion, epoch_number, interval):
    model.train()
    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        ch = output.size()[1] // 2
        rho = torch.zeros_like(data[:, 0, :, :], dtype = torch.cdouble)
        purity_loss = 0

        for i in range(ch):
            rho_real = output[:, i, :, :]
            rho_imag = output[:, ch + i, :, :]
            rho_i = torch.complex(rho_real, rho_imag)
            rho += rho_i
            purity_loss += torch.stack([torch.abs(torch.trace(torch.mm(rho_i[ex], rho_i[ex])) - 1.) for ex in range(len(rho_i))])

        rho /= ch
        rho = torch.stack((rho.real, rho.imag), dim = 1)
        mixed_loss = criterion(rho, data)

        loss = mixed_loss + 0.5 * torch.mean(purity_loss / ch)
        loss.backward()
        optimizer.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def train_sep_class(separator, classifier, device, train_loader, optimizer, criterion, epoch_number, interval):
    classifier.train()
    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        sep_matrices = separator(data)
        metric, _ = sep_met(sep_matrices, data)
        sep_matrices = torch.cat(sep_matrices, dim=1).detach()
        output = classifier(sep_matrices, metric.detach())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def train_from_sep(separator, classifier, device, train_loader, optimizer, criterion, epoch_number, interval, detach = True):
    classifier.train()
    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        sep_matrices = separator(data)
        rho = rho_reconstruction(data, sep_matrices)
        if detach:
            rho = rho.detach()
        new_data = torch.cat((data, rho), dim = 1)
        output = classifier(new_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def test_vector_siamese(model, device, test_loader, criterion, message, confusion_matrix = False, confusion_matrix_dim = None, bipart = 'averaged', negativity_ext = False, low_thresh = 0.5, high_thresh = 0.5, decision_point = 0.5, balanced_acc = False, permute = False):
    model.eval()
    test_loss = 0.
    correct = 0
    if bipart == 'separate':
        correct_lh = np.zeros(test_loader.dataset.bipart_num)
        num_lh = np.zeros(test_loader.dataset.bipart_num)
    else:
        correct_lh = 0
        num_lh = 0
    
    if confusion_matrix or balanced_acc:
        if bipart == 'separate':
            conf_matrix = np.zeros((test_loader.dataset.bipart_num, confusion_matrix_dim, confusion_matrix_dim))
        elif bipart == 'averaged' or bipart == 'single':
            conf_matrix = np.zeros((confusion_matrix_dim, confusion_matrix_dim))
   
    with torch.no_grad():
        for data, target in test_loader:
            if model.dim > data.shape[2]:
                num_qbits = int(round(np.log2(model.dim)))
                data, target = extend_states(data, target, num_qbits)
            data, target = data.to(device), target.to(device) 

            if bipart == 'single':
                ind = np.random.randint(0, len(model.perms))
                data = all_perms(data, ind).double().to(device)
                ind = ind // factorial(int(round(np.log2(model.dim))) - 1)
                output = model(data)
                test_loss += criterion(output[0], torch.unsqueeze(target[:, ind], dim=1)).item() 
            else:
                output = model([data])
                test_loss += criterion(output[0], target).item() 

            prediction = torch.zeros_like(output[0])
            prediction[output[0] > decision_point] = 1

            if negativity_ext:
                prediction[target == 1] = 1            

            if bipart == 'separate':
                correct += (prediction.eq(target)).sum(dim=0).cpu().numpy()

                for i in range(test_loader.dataset.bipart_num):

                    correct_lh[i] += (prediction[:,i][output[0][:,i] < low_thresh].eq(target[:,i][output[0][:,i] < low_thresh])).sum().cpu().numpy()
                    correct_lh[i] += (prediction[:,i][output[0][:,i] > high_thresh].eq(target[:,i][output[0][:,i] > high_thresh])).sum().cpu().numpy()
                    num_lh[i] +=  (prediction[:,i][output[0][:,i] > high_thresh]).shape[0] + (prediction[:,i][output[0][:,i] < low_thresh]).shape[0]
                   
            elif bipart == 'averaged':
                correct += prediction.eq(target).sum().item()

                correct_lh += prediction[output[0] < low_thresh].eq(target[output[0] < low_thresh]).sum().item()
                correct_lh += prediction[output[0] > high_thresh].eq(target[output[0] > high_thresh]).sum().item()

                num_lh += len(output[0] < low_thresh) + len(output[0] > high_thresh)

            elif bipart == 'single':
                correct += prediction.eq(torch.unsqueeze(target[:, ind], dim=1)).sum().item()

                correct_lh += prediction[output[0] < low_thresh].eq(torch.unsqueeze(target[output[0] < low_thresh][:, ind], dim=1)).sum().item()
                correct_lh += prediction[output[0] > high_thresh].eq(torch.unsqueeze(target[output[0] > high_thresh][:, ind], dim=1)).sum().item()

                num_lh += len(output[0] < low_thresh) + len(output[0] > high_thresh)

            if confusion_matrix or balanced_acc:
                if bipart == 'separate':
                    for n in range(test_loader.dataset.bipart_num):
                        for i, j in zip(target[:, n], prediction[:, n]):
                            conf_matrix[n, int(i), int(j)] += 1    
                elif bipart == 'averaged':
                    for i, j in zip(target, prediction):
                        conf_matrix[int(i), int(j)] += 1
                elif bipart == 'single':
                    for i, j in zip(target[:, 0], prediction):
                        conf_matrix[int(i), int(j)] += 1
        
    if balanced_acc:
        if len(conf_matrix.shape) > 2:
            sensitivity = np.array([cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in conf_matrix]) # TP / (TP + FN)
            specifity = np.array([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in conf_matrix]) # TN / (TN + FP)
        else:
            sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
            sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])

        bal_acc = 100.* (sensitivity + specifity) / 2

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    acc_lh = 100. * correct_lh / num_lh

    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
        message, test_loss, correct, len(test_loader.dataset), acc))
    if low_thresh != high_thresh or low_thresh != 0.5:
        print("Accuracy without uncertainity area: {}/{} ({}%)".format(correct_lh, num_lh, acc_lh))
    
    if confusion_matrix:
        print('Confusion matrix:\n{}'.format(conf_matrix))
        if balanced_acc:
            return test_loss, acc, conf_matrix, bal_acc
        else:
            return test_loss, acc, conf_matrix
    
    if low_thresh == high_thresh and low_thresh == 0.5:
        if balanced_acc:
            return test_loss, acc, bal_acc
        else:
            return test_loss, acc
    else:
        if balanced_acc:
            return test_loss, acc, acc_lh, bal_acc
        else:
            return test_loss, acc, acc_lh


def test(model, device, test_loader, criterion, message, confusion_matrix = False, confusion_matrix_dim = None, bipart = False, decision_point = 0.5, balanced_acc = False):
    model.eval()
    model.to(device)
    test_loss = 0.
    correct = 0
    
    if confusion_matrix or balanced_acc:
        if bipart:
            conf_matrix = np.zeros((test_loader.dataset.bipart_num, confusion_matrix_dim, confusion_matrix_dim))
        else:
            conf_matrix = np.zeros((confusion_matrix_dim, confusion_matrix_dim))
   
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) 
            output = model(data)

            test_loss += criterion(output, target).item() 
            prediction = torch.zeros_like(output)
            prediction[output > decision_point] = 1
            #prediction = torch.round(output)              

            if bipart:
                correct += (prediction.eq(target)).sum(dim=0).cpu().numpy()
            else:
                correct += prediction.eq(target).sum().item()

            if confusion_matrix or balanced_acc:
                if bipart:
                    for n in range(test_loader.dataset.bipart_num):
                        for i, j in zip(target[:, n], prediction[:, n]):
                            conf_matrix[n, int(i), int(j)] += 1  
                else:
                    for i, j in zip(target, prediction):
                        conf_matrix[int(i), int(j)] += 1

    if balanced_acc:
        if len(conf_matrix.shape) > 2:
            sensitivity = np.array([cm[0, 0] / (cm[0, 0] + cm[0, 1]) for cm in conf_matrix]) # TP / (TP + FN)
            specifity = np.array([cm[1, 1] / (cm[1, 0] + cm[1, 1]) for cm in conf_matrix]) # TN / (TN + FP)
        else:
            sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
            specifity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])

        bal_acc = 100.* (sensitivity + specifity) / 2

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)

    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
        message, test_loss, correct, len(test_loader.dataset), acc))
    if confusion_matrix:
        print('Confusion matrix:\n{}'.format(conf_matrix))
        if balanced_acc:
            return test_loss, acc, conf_matrix, bal_acc
        else:
            return test_loss, acc, conf_matrix
    
    if balanced_acc:
        return test_loss, bal_acc
    else:
        return test_loss, acc


def test_double(model1, model2, device, test_loader, criterion, message, confusion_matrix = False, confusion_matrix_dim = None, bipart = False, balance_losses = False):
    model1.eval()
    model2.eval()
    test_loss1 = 0.
    test_loss2 = 0.

    correct = 0
    
    if confusion_matrix:
        if bipart:
            conf_matrix = np.zeros((test_loader.dataset.bipart_num, confusion_matrix_dim, confusion_matrix_dim))
        else:
            conf_matrix = np.zeros((confusion_matrix_dim, confusion_matrix_dim))
   
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) 
            output1 = model1(data)
            output2 = model2(data)

            loss1 = criterion(output1, target).item() 
            loss2 = criterion(output2, target).item() 
            test_loss1 += loss1
            test_loss2 += loss2

            if balance_losses:
                bool_matrix =  (loss2.item()/(loss1.item() + loss2.item()))*output1 > (loss1.item()/(loss1.item() + loss2.item()))(1 - output2)
            else:
                bool_matrix = output1 > (1 - output2)     
            prediction = torch.zeros_like(target)
            prediction[bool_matrix] = 1         

            if bipart:
                correct += (prediction.eq(target)).sum(dim=0).cpu().numpy()
            else:
                correct += prediction.eq(target).sum().item()

            if confusion_matrix:
                if bipart:
                    for n in range(test_loader.dataset.bipart_num):
                        for i, j in zip(target[:, n], prediction[:, n]):
                            conf_matrix[n, int(i), int(j)] += 1  
                else:
                    for i, j in zip(target, prediction):
                        conf_matrix[int(i), int(j)] += 1


    test_loss1 /= len(test_loader)
    test_loss2 /= len(test_loader)

    acc = 100. * correct / len(test_loader.dataset)

    print('{}: Average loss1: {:.4f}, Average loss2: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
        message, test_loss1, test_loss2, correct, len(test_loader.dataset), acc))
    if confusion_matrix:
        print('Confusion matrix:\n{}'.format(conf_matrix))
        return test_loss1, test_loss2, acc, conf_matrix
    
    return test_loss1, test_loss2, acc


def test_siamese(model, device, test_loader, criterion, message, confusion_matrix=False, confusion_matrix_dim=None):
    model.eval()
    test_loss = 0.
    correct = 0

    if confusion_matrix:
        conf_matrix = np.zeros((confusion_matrix_dim, confusion_matrix_dim))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data, data)

            test_loss += criterion(output, target).item()
            prediction = torch.round(output)
            correct += prediction.eq(target).sum().item()

            if confusion_matrix:
                for i, j in zip(target, prediction):
                    conf_matrix[int(i), int(j)] += 1

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)

    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        message, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if confusion_matrix:
        print('Confusion matrix:\n{}'.format(conf_matrix))
        return test_loss, acc, conf_matrix

    return test_loss, acc


def test_separator(model, device, test_loader, criterion, message, threshold = 0.1, use_noise = False):
    model.eval()
    test_loss = 0.
    correct = 0
   
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if use_noise:
                noise_ch = model.input_channels - 2
                noise = torch.randn(data.size()[0], noise_ch, data.size()[2], data.size()[3]).to(device)
                data = torch.cat((data, noise), dim = 1)

            output = model(data)
            ch = output[0].size()[1] // 2
            rho = torch.zeros_like(data[:, 0, :, :], dtype = torch.cdouble)
            trace_loss = 0
            hermitian_loss = 0

            for i in range(ch):
                dms = output[0]
                rho_real = dms[:, i, :, :]
                rho_imag = dms[:, ch + i, :, :]
                rho_i = torch.complex(rho_real, rho_imag)
            
                for j in range(1, len(output)):
                    dms = output[j]
                    ch = dms.size()[1] // 2
                    rho_real = dms[:, i, :, :]
                    rho_imag = dms[:, ch + i, :, :]
                    rho_j = torch.complex(rho_real, rho_imag)    
                    rho_i = torch.stack([torch.kron(rho_i[k], rho_j[k]) for k in range(len(rho_i))])
                rho += rho_i

            rho = rho / ch
            rho = torch.stack((rho.real, rho.imag), dim = 1)
            loss = criterion(rho, data[:, :2, :, :])
            rho_diff = torch.abs(rho - data)

            for ex in range(data.size()[0]):
                if torch.mean(rho_diff[ex]) < threshold:
                    correct += 1

            test_loss += loss

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)

    print('{}: Average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(message, test_loss, correct, len(test_loader.dataset), acc))

    return test_loss, acc


def test_separator_as_classifier(model, device, test_loader, criterion, message, threshold, use_noise=False, confusion_matrix = False):
    model.eval()
    test_loss = 0.
    true_test_loss = 0.
    false_test_loss = 0.
    correct = 0

    if confusion_matrix:
        conf_matrix = np.zeros((2, 2))
   
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) 

            if use_noise:
                noise_ch = model.input_channels - 2
                noise = torch.randn(data.size()[0], noise_ch, data.size()[2], data.size()[3]).to(device)
                new_data = torch.cat((data, noise), dim = 1)

            else:
                new_data = data
                  
            output = model(new_data)
            ch = output[0].size()[1] // 2
            rho = torch.zeros_like(data[:,0,:,:], dtype = torch.cdouble)

            for i in range(ch):
                dms = output[0]
                rho_real = dms[:, i, :, :]
                rho_imag = dms[:, ch + i, :, :]
                rho_i = torch.complex(rho_real, rho_imag)
                
                for j in range(1, len(output)):
                    dms = output[j]
                    ch = dms.size()[1] // 2
                    rho_real = dms[:, i, :, :]
                    rho_imag = dms[:, ch + i, :, :]
                    rho_j = torch.complex(rho_real, rho_imag) 
                    rho_i = torch.stack([torch.kron(rho_i[k], rho_j[k]) for k in range(len(rho_i))])
                        
                rho += rho_i
                
            rho = rho / ch
              
            if criterion == 'bures':
                data_complex = torch.complex(data[:, 0, :, :], data[:, 1, :, :])
                loss = torch.stack([2*torch.abs(1 - torch.sqrt(torch_fidelity(rho[i], data_complex[i]))) for i in range(data_complex.size()[0])])
            else:
                rho = torch.stack((rho.real, rho.imag), dim = 1)
                loss = torch.mean(criterion(rho, data), dim=(1,2,3))

            prediction = torch.ones_like(target)

            for ex in range(data.size()[0]):
                if loss[ex] < threshold:
                    prediction[ex] = 0
                
                if confusion_matrix:
                    if target[ex] == 1:
                        false_test_loss += loss[ex]  
                    else:
                        true_test_loss += loss[ex]
                    
            correct += prediction.eq(target).sum().item()
            test_loss += torch.mean(loss).item()

            if confusion_matrix:
                for i, j in zip(target, prediction):
                    conf_matrix[int(i), int(j)] += 1

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)

    if confusion_matrix:
        true_test_loss /= (conf_matrix[0, 0] + conf_matrix[0, 1] + 1.e-7)
        false_test_loss /= (conf_matrix[1, 0] + conf_matrix[1, 1] + 1.e-7)


    print('{}: Average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(message, test_loss, correct, len(test_loader.dataset), acc))
    if confusion_matrix:
        print('Confusion matrix:\n{}'.format(conf_matrix))
        return (test_loss, true_test_loss, false_test_loss), acc, conf_matrix

    return test_loss, acc


def separator_predict(model, device, data, threshold, criterion = 'L1', return_loss = False):
    data = data.to(device)
    output = model(data)
    ch = output[0].size()[1] // 2
    rho = torch.zeros_like(data[:,0,:,:], dtype = torch.cdouble)

    for i in range(ch):
        dms = output[0]
        rho_real = dms[:, i, :, :]
        rho_imag = dms[:, ch + i, :, :]
        rho_i = torch.complex(rho_real, rho_imag)
                
        for j in range(1, len(output)):
            dms = output[j]
            ch = dms.size()[1] // 2
            rho_real = dms[:, i, :, :]
            rho_imag = dms[:, ch + i, :, :]
            rho_j = torch.complex(rho_real, rho_imag) 
            rho_i = torch.stack([torch.kron(rho_i[k], rho_j[k]) for k in range(len(rho_i))])
                        
        rho += rho_i
                
    rho = rho / ch

    if criterion == 'bures':
        data_complex = torch.complex(data[:, 0, :, :], data[:, 1, :, :])
        loss = torch.stack([2*(torch.abs(1 - torch.sqrt(torch_fidelity(rho[i], data_complex[i])))) for i in range(data_complex.size()[0])])
    else:
        criterion = nn.L1Loss(reduction='none')
        rho = torch.stack((rho.real, rho.imag), dim = 1)
        loss = torch.mean(criterion(rho, data), dim=(1,2,3))

    prediction = torch.ones(data.size()[0]).to(device)

    for ex in range(data.size()[0]):
        if loss[ex] < threshold:
            prediction[ex] = 0

    if return_loss:
        return prediction, loss
    else:
        return prediction
    

def test_separator_bipart(model, device, test_loader, criterion, message, threshold = 0.1, specific_bipartition = None):
    model.eval()
    test_loss = 0.
    correct = torch.zeros(model.qbits_num).to(device)
   
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
        
            q_fac = factorial(model.qbits_num - 1)
            if specific_bipartition == None:
                inds = [q * q_fac for q in range(model.qbits_num)]
            else:
                inds = [q_fac*specific_bipartition]
            biparts_num = len(inds)
            perm_data = all_perms(data, inds).double().to(device)

            diff_loss = torch.zeros(biparts_num).to(device)
            trace_loss = torch.zeros(biparts_num).to(device)
            hermitian_loss = torch.zeros(biparts_num).to(device)

            rho_diff = torch.zeros((biparts_num, *data.size())).to(device)

            for k in range(biparts_num):
                output = model(perm_data[k])
                ch = output[0].size()[1] // 2

                rho_sep1 = output[0]
                rho_sep2 = output[1]

                rho = torch.zeros_like(data[:, 0, :, :], dtype = torch.cdouble)
            
                for i in range(ch):
                    rho1_real = rho_sep1[:, i, :, :]
                    rho1_imag = rho_sep1[:, ch + i, :, :]
                    rho1_i = torch.complex(rho1_real, rho1_imag)
                    trace_diffs1 = torch.stack([torch.abs(torch.trace(rho1_i[ex]) - 1.) for ex in range(len(rho1_i))])

                    rho2_real = rho_sep2[:, i, :, :]
                    rho2_imag = rho_sep2[:, ch + i, :, :]
                    rho2_i = torch.complex(rho2_real, rho2_imag)
                    trace_diffs2 = torch.stack([torch.abs(torch.trace(rho2_i[ex]) - 1.) for ex in range(len(rho2_i))])

                    trace_loss[k] += torch.mean(trace_diffs1) + torch.mean(trace_diffs2)
                    hermitian_loss[k] += torch.mean(torch.abs(torch.conj(torch.transpose(rho1_i, -1, -2)) - rho1_i))  
                    hermitian_loss[k] += torch.mean(torch.abs(torch.conj(torch.transpose(rho2_i, -1, -2)) - rho2_i))   

                    rho_i = torch.stack([torch.kron(rho1_i[ex], rho2_i[ex]) for ex in range(len(rho1_i))])
                    rho += rho_i

                rho = rho / ch
                rho = torch.stack((rho.real, rho.imag), dim = 1)
                diff_loss[k] = criterion(rho, perm_data[k])
                hermitian_loss[k] /= 2*ch
                trace_loss[k] /= 2*ch
                rho_diff[k] = torch.abs(rho - perm_data[k])


            for ex in range(data.size()[0]):
                for k in range(biparts_num):
                    if torch.mean(rho_diff[k, ex]) < threshold:
                        correct[k] += 1

            loss = torch.mean(diff_loss) + 0.1 * torch.mean(trace_loss) + 0.1 * torch.mean(hermitian_loss)
            test_loss += loss

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)

    print('{}: Average loss: {:.4f}, accuracy: {}/{} ({}%)\n'.format(message, test_loss, correct.cpu().numpy(), len(test_loader.dataset), acc.cpu().numpy()))

    return test_loss, acc.cpu().numpy()


def test_purificator(model, device, test_loader, criterion, message, threshold = 0.1):
    model.eval()
    test_loss = 0.
    correct = 0
   
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
        
            output = model(data)
            ch = output.size()[1] // 2
            rho = torch.zeros_like(data[:, 0, :, :], dtype = torch.cdouble)
            purity_loss = 0

            for i in range(ch):
                rho_real = output[:, i, :, :]
                rho_imag = output[:, ch + i, :, :]
                rho_i = torch.complex(rho_real, rho_imag)
                rho += rho_i
                purity_loss += torch.stack([torch.abs(torch.trace(torch.mm(rho_i[ex], rho_i[ex])) - 1.) for ex in range(len(rho_i))])

            rho /= ch
            rho = torch.stack((rho.real, rho.imag), dim = 1)
            mixed_loss = torch.mean(torch.mean(torch.mean(torch.abs(rho - data), dim = 1), dim = 1), dim = 1)
            loss = mixed_loss + 0.5 * purity_loss / ch

            for ex in range(data.size()[0]):
                if loss[ex] < threshold:
                    correct += 1

            test_loss += torch.mean(loss)

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)

    print('{}: Average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(message, test_loss, correct, len(test_loader.dataset), acc))

    return test_loss, acc

def test_sep_class(separator, classifier, device, test_loader, criterion, message, confusion_matrix = False, confusion_matrix_dim = None, bipart = False):
    classifier.eval()
    test_loss = 0.
    correct = 0
    
    if confusion_matrix:
        if bipart:
            conf_matrix = np.zeros((test_loader.dataset.bipart_num, confusion_matrix_dim, confusion_matrix_dim))
        else:
            conf_matrix = np.zeros((confusion_matrix_dim, confusion_matrix_dim))
   
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) 
            sep_matrices = separator(data)
            metric, _ = sep_met(sep_matrices, data)
            sep_matrices = torch.cat(sep_matrices, dim=1).detach()
            output = classifier(sep_matrices, metric.detach())

            test_loss += criterion(output, target).item() 
            prediction = torch.round(output)              

            if bipart:
                correct += (prediction.eq(target)).sum(dim=0).cpu().numpy()
            else:
                correct += prediction.eq(target).sum().item()

            if confusion_matrix:
                if bipart:
                    for n in range(test_loader.dataset.bipart_num):
                        for i, j in zip(target[:, n], prediction[:, n]):
                            conf_matrix[n, int(i), int(j)] += 1    
                else:
                    for i, j in zip(target, prediction):
                        conf_matrix[int(i), int(j)] += 1


    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)

    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
        message, test_loss, correct, len(test_loader.dataset), acc))
    if confusion_matrix:
        print('Confusion matrix:\n{}'.format(conf_matrix))
        return test_loss, acc, conf_matrix

    return test_loss, acc


def test_from_sep(separator, classifier, device, test_loader, criterion, message, confusion_matrix = False, confusion_matrix_dim = None, bipart = False):
    classifier.eval()
    test_loss = 0.
    correct = 0
    
    if confusion_matrix:
        if bipart:
            conf_matrix = np.zeros((test_loader.dataset.bipart_num, confusion_matrix_dim, confusion_matrix_dim))
        else:
            conf_matrix = np.zeros((confusion_matrix_dim, confusion_matrix_dim))
   
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) 
            sep_matrices = separator(data)
            rho = rho_reconstruction(data, sep_matrices)
            new_data = torch.cat((data, rho), dim = 1)
            output = classifier(new_data)

            test_loss += criterion(output, target).item() 
            prediction = torch.round(output)              

            if bipart:
                correct += (prediction.eq(target)).sum(dim=0).cpu().numpy()
            else:
                correct += prediction.eq(target).sum().item()

            if confusion_matrix:
                if bipart:
                    for n in range(test_loader.dataset.bipart_num):
                        for i, j in zip(target[:, n], prediction[:, n]):
                            conf_matrix[n, int(i), int(j)] += 1    
                else:
                    for i, j in zip(target, prediction):
                        conf_matrix[int(i), int(j)] += 1


    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)

    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
        message, test_loss, correct, len(test_loader.dataset), acc))
    if confusion_matrix:
        print('Confusion matrix:\n{}'.format(conf_matrix))
        return test_loss, acc, conf_matrix

    return test_loss, acc


def test_reg(model, device, test_loader, criterion, message, print_examples = False, threshold = 0.001, bipart = False):
    model.eval()
    test_loss = 0.
    if bipart:
        correct = np.zeros(test_loader.dataset.bipart_num)
    else:
        correct = 0

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device) 
            output = model(data)

            test_loss += criterion(output, target).item() 

            if bipart:
                correct += ((output > threshold).eq(target > 0.001)).sum(dim=0).cpu().numpy()
            else:
                correct += ((output > threshold).eq(target > 0.001)).sum().item()

            if print_examples:
                ex = np.random.choice(output.shape[0])
                print("Example {}: output: {}, target: {}".format(idx, output[ex, 0], target[ex, 0]))


    test_loss /= len(test_loader)
    acc = 100 * correct / len(test_loader.dataset)
    print('{}: Average loss: {:.4f}, Accuracy: {}%\n'.format(message, test_loss, acc))
    
    return test_loss, acc


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