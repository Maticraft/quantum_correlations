import numpy as np
from commons.pytorch_utils import torch_fidelity


import torch


def separator_loss(data, output, criterion):
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


def symmetry_loss(output_1, output_2, qubits_for_sym):
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