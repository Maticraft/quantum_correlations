from commons.pytorch_utils import rho_reconstruction, sep_met


import numpy as np
import torch


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