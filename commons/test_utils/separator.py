from math import factorial
import numpy as np
import torch
import torch.nn as nn

from commons.pytorch_utils import all_perms, torch_fidelity


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
            loss = calculate_separator_loss(model, device, criterion, use_noise, data, target)

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


def calculate_separator_loss(model, device, criterion, use_noise, data, target):
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

    return loss


def test_multi_separator_as_classifier(models, device, test_loader, criterion, message, last_threshold, prev_separator_thresholds = [], use_noise=False, confusion_matrix = False):
    test_loss = 0.
    true_test_loss = 0.
    false_test_loss = 0.
    correct = 0

    assert len(prev_separator_thresholds) == len(models), "Number of thresholds must be equal to number of models"

    if confusion_matrix:
        conf_matrix = np.zeros((2, 2))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            losses = []
            for model in models:
                model.eval()
                loss = calculate_separator_loss(model, device, criterion, use_noise, data, target)
                losses.append(loss)
            
            loss = torch.stack(losses, dim=-1)
            prediction = torch.ones_like(target)
            for ex in range(data.size()[0]):
                for i, prev_separator_threshold in enumerate(prev_separator_thresholds[:-1]):
                    if loss[ex, i].item() < prev_separator_threshold:
                        prediction[ex] = 0
                if loss[ex, -1].item() < last_threshold:
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