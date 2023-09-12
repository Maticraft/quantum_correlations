import torch

from commons.loss.regularization import regularization_loss


def train_semi_supervised(teacher_model, student_model, device, train_loader, optimizer, criterion, epoch_number, interval = 100, regularizer_loss_rate = 0.5, add_noise=False):
    teacher_model.eval()
    teacher_model.to(device)
    student_model.train()
    student_model.to(device)
    total_supervised_loss = 0.
    total_reg_loss = 0.

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        pseudo_labels = torch.round(teacher_model(data).detach())

        optimizer.zero_grad()
        output = student_model(data)
        loss = criterion(output, pseudo_labels)
        total_supervised_loss += loss.item()

        reg_loss = regularization_loss(data, output, student_model, device, criterion, add_noise)
        total_reg_loss += reg_loss.item()
        loss += reg_loss * regularizer_loss_rate

        loss.backward()
        optimizer.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    total_supervised_loss /= len(train_loader)
    total_reg_loss /= len(train_loader)

    print('\nTrain set: Supervised loss: {:.4f}, Regularizer loss {:.4f}'.format(total_supervised_loss, total_reg_loss))
    return total_supervised_loss, total_reg_loss