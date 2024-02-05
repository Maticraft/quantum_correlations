from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

def test_matrix_reconstructor(model, device, test_loader, criterion, message):
    model.eval()
    model.to(device)
    test_loss = 0
    with torch.no_grad():
        for rho, measurement, _ in tqdm(test_loader, desc=f'{message}'):
            rho, measurement = rho.to(device), measurement.to(device)
            predicted_rho = model(measurement)
            loss = criterion(predicted_rho, rho)
            test_loss += loss.item()
        test_loss /= len(test_loader)
    return test_loss


def test(model, device, test_loader, criterion, message, confusion_matrix = False, confusion_matrix_dim = 2, bipart = False, decision_point = 0.5, balanced_acc = False):
    model.eval()
    model.to(device)
    test_loss = 0.
    true_prob = 0.
    false_prob = 0.
    correct = 0

    if confusion_matrix or balanced_acc:
        if bipart:
            conf_matrix = np.zeros((test_loader.dataset.bipart_num, confusion_matrix_dim, confusion_matrix_dim))
        else:
            conf_matrix = np.zeros((confusion_matrix_dim, confusion_matrix_dim))

    with torch.no_grad():
        for _, measurement, target in tqdm(test_loader, desc='Testing model...'):
            data, target = measurement.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            prediction = torch.zeros_like(output)
            prediction[output > decision_point] = 1

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
                
                false_prob += output[target == 0].sum().item()
                true_prob += output[target == 1].sum().item()

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
        if len(conf_matrix.shape) > 2:
            false_prob /= conf_matrix[:, 0, 0].sum() + conf_matrix[:, 0, 1].sum()
            true_prob /= conf_matrix[:, 1, 1].sum() + conf_matrix[:, 1, 0].sum()
        else:
            false_prob /= conf_matrix[0, 0] + conf_matrix[0, 1]
            true_prob /= conf_matrix[1, 1] + conf_matrix[1, 0]

        if balanced_acc:
            return test_loss, acc, conf_matrix, true_prob, false_prob, bal_acc 
        else:
            return test_loss, acc, conf_matrix, true_prob, false_prob

    if balanced_acc:
        return test_loss, acc, bal_acc
    else:
        return test_loss, acc