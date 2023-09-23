import numpy as np
import torch


def test(model, device, test_loader, criterion, message, confusion_matrix = False, confusion_matrix_dim = 2, bipart = False, decision_point = 0.5, balanced_acc = False):
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