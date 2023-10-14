import torch

def train_ensemble(model, device, train_loader, optimizer, criterion, epoch_number, interval):
    model.to(device)
    model.train()

    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        outputs, weights = model(data, return_all=True)
        target = target.unsqueeze(1).expand_as(outputs)
        ensemble_loss = criterion(outputs, target)
        loss = torch.mean(ensemble_loss * weights.unsqueeze(-1))

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