import torch

def train_ensemble(model, device, train_loader, optimizer, epoch_number, interval):
    model.to(device)
    model.train()

    train_loss = 0.

    bce_loss = torch.nn.BCELoss(reduction='none')
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        outputs, weights = model(data, return_all=True)
        target = target.unsqueeze(1).expand_as(outputs)
        ensemble_loss = bce_loss(outputs, target)
        best_submodels_ids = torch.argmin(ensemble_loss.mean(-1), dim=-1, keepdim=True)
        weights_target = torch.zeros_like(weights)
        weights_target.scatter_(1, best_submodels_ids, 1)
        weights_loss = cross_entropy_loss(weights, weights_target)
        # loss = torch.mean(ensemble_loss * weights.unsqueeze(-1)) + weights_loss
        ensemble_loss = torch.gather(ensemble_loss, 1, best_submodels_ids.unsqueeze(-1).expand(-1, -1, ensemble_loss.shape[-1]))

        loss = torch.mean(ensemble_loss) + weights_loss

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