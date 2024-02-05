from tqdm import tqdm

import torch.nn as nn

def train_matrix_reconstructor(model, device, train_loader, optimizer, epoch_number, interval: int = 100, criterion = nn.MSELoss()):
    model.train()
    model.to(device)
    train_loss = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch_number}')
    for batch_idx, (rho, measurement, _) in pbar:
        rho, measurement = rho.to(device), measurement.to(device)
        optimizer.zero_grad()
        predicted_rho = model(measurement)
        loss = criterion(predicted_rho, rho)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % interval == 0:
            pbar.set_postfix({'loss': loss.item()})
    train_loss /= len(train_loader)
    return train_loss


def train(model, device, train_loader, optimizer, criterion, epoch_number, interval):
    model.train()
    model.to(device)
    train_loss = 0.

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch_number}')
    for batch_idx, (rho, measurement, target) in pbar:
        measurement, target = measurement.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(measurement)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % interval == 0:
            pbar.set_postfix({'loss': loss.item()})

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss