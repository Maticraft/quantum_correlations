import torch.nn as nn
from commons.pytorch_utils import all_perms, loc_op, loc_op_circ, pure_rep


import numpy as np
import torch


from math import factorial, log2

from commons.train.purificator import train_purificator


def train_vector_siamese(model, device, train_loader, optimizer, criterion, epoch_number, interval, loc_op_flag = False, reduced_perms_num = None, pure_representation = False, biparts = 'separate'):
    model.train()
    train_loss = 0.

    if epoch_number <= 10:
        lambda_1 = 0
    else:
        lambda_1 = epoch_number/20

    for batch_idx, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()

        if reduced_perms_num != None:
            if biparts == 'single':
                inds = [0] + list(np.random.choice(np.arange(1, factorial(int(round(np.log2(model.dim))) - 1)), size = reduced_perms_num, replace=False))
            else:
                inds = [0] + list(np.random.choice(np.arange(1, len(model.perms)), size = reduced_perms_num, replace=False))
            perm_data = all_perms(data, inds).double().to(device)
        else:
            perm_data = all_perms(data).double().to(device)

        if pure_representation and epoch_number > 10:
            pcm = pure_rep(target, int(log2(model.dim))).double().to(device)
            perm_data = torch.cat((perm_data, torch.torch.unsqueeze(pcm, dim=0)), dim = 0)

        if loc_op_flag:
            loc_op_data = loc_op_circ(data).double().to(device)
            outputs = model(torch.cat((perm_data, torch.unsqueeze(loc_op_data, dim=0)), dim=0))
        else:
            outputs = model(perm_data)

        target = target.to(device)

        losses_perm = []

        # separate outputs for different bipartitions
        if biparts == 'separate':
            loss_std = criterion(outputs[0], target)

            for (i1, j1), (i2, j2) in model.matching_indices:
                if reduced_perms_num != None:
                    if j1 in inds and j2 in inds:
                        losses_perm.append(torch.mean(torch.abs(outputs[inds.index(j1),:,i1] - outputs[inds.index(j2),:,i2])))
                else:
                    losses_perm.append(torch.mean(torch.abs(outputs[j1,:,i1] - outputs[j2,:,i2])))

        elif biparts == 'averaged': # single output averaged bipartitions
            loss_std = criterion(outputs[0], target)

            for j1 in range(len(outputs)):
                for j2 in range((len(outputs))):
                    if reduced_perms_num != None:
                        if j1 in inds and j2 in inds:
                            losses_perm.append(torch.mean(torch.abs(outputs[inds.index(j1)] - outputs[inds.index(j2)])))
                    else:
                        losses_perm.append(torch.mean(torch.abs(outputs[j1] - outputs[j2])))

        elif biparts == 'single':
            loss_std = criterion(outputs[0], torch.unsqueeze(target[:,0], dim=1))
            for (i1, j1), (i2, j2) in model.matching_indices:
                if i1 == i2 and i1 == 0:
                    if reduced_perms_num != None:
                        if j1 in inds and j2 in inds:
                            losses_perm.append(torch.mean(torch.abs(outputs[inds.index(j1),:] - outputs[inds.index(j2),:])))
                    else:
                        losses_perm.append(torch.mean(torch.abs(outputs[j1,:] - outputs[j2,:])))

        else:
            raise ValueError("Wrong biparts mode")


        if loc_op_flag:
            loss_loc_op1 = torch.mean(torch.abs(outputs[0] - outputs[outputs.size()[0] - 1]))

            if pure_representation and epoch_number > 10:
                loss_pr = torch.mean(torch.abs(outputs[0] - outputs[outputs.size()[0] - 2]))
                loss = loss_std + 0.5*torch.mean(torch.stack(losses_perm)) + 0.5*loss_loc_op1 + lambda_1*loss_pr

            else:
                loss = loss_std + 0.5*torch.mean(torch.stack(losses_perm)) + 0.5*loss_loc_op1 #+ 0.5*loss_loc_op2
        else:
            if pure_representation and epoch_number > 10:
                loss_pr = torch.mean(torch.abs(outputs[0] - outputs[outputs.size()[0] - 1]))
                loss = loss_std + 0.5*torch.mean(torch.stack(losses_perm)) + lambda_1*loss_pr
            else:
                loss = loss_std + 0.5*torch.mean(torch.stack(losses_perm))

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


def train_vector_siamese_with_purificator(model_siam, model_pure, device, train_loader, optimizer_siam, optimizer_pure, epoch_number, interval, loc_op_flag = False, reduced_perms_num = None):
    print("Epoch: {} - Purificator training".format(epoch_number))
    pure_loss = train_purificator(model_pure, device, train_loader, optimizer_pure, nn.L1Loss(), epoch_number, interval)

    print("Epoch: {} - Siamese training".format(epoch_number))
    model_siam.train()
    train_loss = pure_loss

    for batch_idx, (data, target) in enumerate(train_loader):

        target = target.to(device)

        if reduced_perms_num != None:
            inds = [0] + list(np.random.choice(np.arange(1, len(model_siam.perms)), size = reduced_perms_num, replace=False))
            perm_data = all_perms(data, inds).double().to(device)
        else:
            perm_data = all_perms(data).double().to(device)

        optimizer_siam.zero_grad()
        if loc_op_flag:
            loc_op_data = loc_op_circ(data).double().to(device)
            perm_data = torch.cat((perm_data, torch.unsqueeze(loc_op_data, dim=0)), dim=0)

        purified_data = torch.tensor([model_pure(x) for x in perm_data])
        outputs = model_siam(purified_data)

        criterion = nn.BCELoss()
        loss_std = criterion(outputs[0], target)
        losses_perm = []

        for (i1,j1), (i2, j2) in model_siam.matching_indices:
            if reduced_perms_num != None:
                if j1 in inds and j2 in inds:
                    losses_perm.append(torch.mean(torch.abs(outputs[inds.index(j1),:,i1] - outputs[inds.index(j2),:,i2])))
            else:
                losses_perm.append(torch.mean(torch.abs(outputs[j1,:,i1] - outputs[j2,:,i2])))

        if loc_op_flag:
            loss_loc_op1 = torch.mean(torch.abs(outputs[0] - outputs[outputs.size()[0] - 1]))
            loss = loss_std + 0.5*torch.mean(torch.stack(losses_perm)) + 0.5*loss_loc_op1
        else:
            loss = loss_std + 0.5*torch.mean(torch.stack(losses_perm))

        loss.backward()
        optimizer_siam.step()

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss


def train_siamese(model, device, train_loader, optimizer, criterion, epoch_number, interval):
    model.train()
    train_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):

        data1, target = data.to(device), target.to(device)
        data2 = loc_op(data).to(device)

        optimizer.zero_grad()
        output1, output2 = model(data1, data2)
        loss1 = criterion(output1, target)
        loss2 = torch.mean(torch.abs(output1 - output2))
        loss = loss1 + loss2
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