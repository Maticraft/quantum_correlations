import numpy as np
import torch
import torch.nn as nn

from commons.test_utils.separator import calculate_separator_loss

def test_multi_classifier(models, separator, thresholds, device, test_loader, criterion, message, confusion_matrix = False, confusion_matrix_dim = None, bipart = False, decision_point = 0.5, balanced_acc = False):
    test_loss = 0.
    correct = 0
    separator.eval()
    separator.to(device)
    for model in models:
        model.eval()
        model.to(device)

    if confusion_matrix or balanced_acc:
        if bipart:
            conf_matrix = np.zeros((test_loader.dataset.bipart_num, confusion_matrix_dim, confusion_matrix_dim))
        else:
            conf_matrix = np.zeros((confusion_matrix_dim, confusion_matrix_dim))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            separator_loss = calculate_separator_loss(separator, device, criterion=nn.L1Loss(reduction='none'), use_noise=False, data=data)
                        
            models_ids = [get_model_idx_from_loss_thresholds(separator_loss[i].item(), thresholds) for i in range(len(separator_loss))]
            models_outputs = [models[model_id](data[i].unsqueeze(dim=0)) for i, model_id in enumerate(models_ids)]
            output = torch.cat(models_outputs, dim=0)

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
    

def get_model_idx_from_loss_thresholds(loss, thresholds):
    for i in range(len(thresholds) - 1):
        if thresholds[i] < loss < thresholds[i+1]:
            return i
    return len(thresholds) - 2
