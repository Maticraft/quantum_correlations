from math import factorial
from commons.pytorch_utils import all_perms, amplitude_symmetry_dm, get_separator_loss, get_symmetry_loss


import numpy as np
import torch


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