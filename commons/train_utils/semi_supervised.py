import numpy as np
import torch

from commons.data.generation_functions import generate_parametrized_qs
from commons.data.circuit_ops import local_randomize_matrix
from commons.pytorch_utils import loc_op_circ


def regularization_loss(data, output, model, device, criterion, add_noise=False, eps=0.01):
    loc_op_data = loc_op_circ(data).double().to(device)
    loc_op_output = model(loc_op_data)
    reg_loss = criterion(loc_op_output, output)

    if add_noise:
        num_qubits = int(np.log2(data.shape[-1]))
        noise_data = [construct_simple_separable_matrix(num_qubits) for _ in range(data.shape[0])]
        noise_data = torch.stack(noise_data, dim=0)

        new_data = (1-eps)*data + eps*noise_data
        new_data = new_data.to(device)
        new_output = model(new_data)
        reg_loss += criterion(new_output, output)
    
    return reg_loss


def construct_simple_separable_matrix(num_qubits):
    a = np.random.uniform(size=num_qubits)
    c = np.random.uniform(size=num_qubits)
    random_rho = generate_parametrized_qs(num_qubits, a, c, fi2=0, fi3=0)
    rho = local_randomize_matrix(np.arange(num_qubits), random_rho, 2)
    t_rho = torch.from_numpy(rho.data)
    return torch.stack([t_rho.real, t_rho.imag], dim=0)


def train_semi_supervised(teacher_model, student_model, device, train_loader, optimizer, criterion, epoch_number, interval = 100, regularizer_loss_rate = 0.5, add_noise=False):
    teacher_model.eval()
    teacher_model.to(device)
    student_model.train()
    student_model.to(device)
    train_loss = 0.

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        pseudo_labels = torch.round(teacher_model(data).detach())

        optimizer.zero_grad()
        output = student_model(data)
        loss = criterion(output, pseudo_labels)
        reg_loss = regularization_loss(data, output, student_model, device, criterion, add_noise)
        loss += reg_loss * regularizer_loss_rate

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