import numpy as np
import torch
from qiskit.quantum_info import DensityMatrix, partial_trace

from commons.pytorch_utils import torch_fidelity
import torch.nn as nn


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