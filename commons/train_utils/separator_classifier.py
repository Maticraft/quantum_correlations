from commons.pytorch_utils import rho_reconstruction, sep_met

import torch


def train_sep_class(separator, classifier, device, train_loader, optimizer, criterion, epoch_number, interval):
    classifier.train()
    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        sep_matrices = separator(data)
        metric, _ = sep_met(sep_matrices, data)
        sep_matrices = torch.cat(sep_matrices, dim=1).detach()
        output = classifier(sep_matrices, metric.detach())
        loss = criterion(output, target)
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


def train_from_sep(separator, classifier, device, train_loader, optimizer, criterion, epoch_number, interval, detach = True):
    classifier.train()
    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        sep_matrices = separator(data)
        rho = rho_reconstruction(data, sep_matrices)
        if detach:
            rho = rho.detach()
        new_data = torch.cat((data, rho), dim = 1)
        output = classifier(new_data)
        loss = criterion(output, target)
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