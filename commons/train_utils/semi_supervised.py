import torch

from commons.loss.regularization import regularization_loss


def train_semi_supervised(teacher_model, student_model, device, train_loader, optimizer, criterion, epoch_number, interval = 100, regularizer_loss_rate = 0.5, add_noise=False):
    teacher_model.eval()
    teacher_model.to(device)
    student_model.train()
    student_model.to(device)
    train_loss = 0.

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        pseudo_labels = torch.round(teacher_model(data).detach())

        optimizer.zero_grad()
        output = student_model(data)
        loss = criterion(output, pseudo_labels)
        reg_loss = regularization_loss(data, output, student_model, device, criterion, add_noise)
        loss += reg_loss * regularizer_loss_rate

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