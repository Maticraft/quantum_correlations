from commons.pytorch_utils import all_perms, extend_states


import numpy as np
import torch


from math import factorial


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