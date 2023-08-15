import torch


def train_purificator(model, device, train_loader, optimizer, criterion, epoch_number, interval):
    model.train()
    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        ch = output.size()[1] // 2
        rho = torch.zeros_like(data[:, 0, :, :], dtype = torch.cdouble)
        purity_loss = 0

        for i in range(ch):
            rho_real = output[:, i, :, :]
            rho_imag = output[:, ch + i, :, :]
            rho_i = torch.complex(rho_real, rho_imag)
            rho += rho_i
            purity_loss += torch.stack([torch.abs(torch.trace(torch.mm(rho_i[ex], rho_i[ex])) - 1.) for ex in range(len(rho_i))])

        rho /= ch
        rho = torch.stack((rho.real, rho.imag), dim = 1)
        mixed_loss = criterion(rho, data)

        loss = mixed_loss + 0.5 * torch.mean(purity_loss / ch)
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