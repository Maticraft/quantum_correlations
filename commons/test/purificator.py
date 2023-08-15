import torch


def test_purificator(model, device, test_loader, criterion, message, threshold = 0.1):
    model.eval()
    test_loss = 0.
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

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
            mixed_loss = torch.mean(torch.mean(torch.mean(torch.abs(rho - data), dim = 1), dim = 1), dim = 1)
            loss = mixed_loss + 0.5 * purity_loss / ch

            for ex in range(data.size()[0]):
                if loss[ex] < threshold:
                    correct += 1

            test_loss += torch.mean(loss)

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)

    print('{}: Average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(message, test_loss, correct, len(test_loader.dataset), acc))

    return test_loss, acc