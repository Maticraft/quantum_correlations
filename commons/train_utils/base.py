def train(model, device, train_loader, optimizer, criterion, epoch_number, interval, target_to_filter = None):
    model.train()
    model.to(device)
    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[target != target_to_filter], target[target != target_to_filter])
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


def train_double(model1, model2, device, train_loader, optimizer1, optimizer2, criterion, epoch_number, interval, balance_losses = False, thresh = 0.1):
    model1.train()
    model2.train()

    train_loss1 = 0.
    train_loss2 = 0.


    for batch_idx, (data, target1, target2) in enumerate(train_loader):

        data, target1, target2 = data.to(device), target1.to(device), target2.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        output1 = model1(data)
        loss1 = criterion(output1, target1)

        output2 = model2(data)
        loss2 = criterion(output2, target2)

        if balance_losses and (loss1.item() + thresh) < loss2.item():
            loss2.backward()
            optimizer2.step()
        elif balance_losses and (loss2.item() + thresh) < loss1.item():
            loss1.backward()
            optimizer1.step()
        else:
            loss1.backward()
            optimizer1.step()
            loss2.backward()
            optimizer2.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss1.item(), loss2.item()))

        train_loss1 += loss1.item()
        train_loss2 += loss2.item()


    train_loss1 /= len(train_loader)
    train_loss2 /= len(train_loader)


    print('\nTrain set:\n Average loss1: {:.4f}\nAverage loss2: {:.4f}'.format(train_loss1, train_loss2))
    return train_loss1, train_loss2