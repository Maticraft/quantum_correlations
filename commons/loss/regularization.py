from commons.pytorch_utils import construct_simple_separable_matrix, loc_op_circ

import numpy as np
import torch


def regularization_loss(data, output, model, device, criterion, add_noise=False, eps=1.e-4):
    loc_op_data = loc_op_circ(data).double().to(device)
    loc_op_output = model(loc_op_data)
    reg_loss = criterion(loc_op_output, output)

    if add_noise:
        num_qubits = int(np.log2(data.shape[-1]))
        noise_data = [construct_simple_separable_matrix(num_qubits) for _ in range(data.shape[0])]
        noise_data = torch.stack(noise_data, dim=0).to(device)

        new_data = (1-eps)*data + eps*noise_data
        new_data = new_data.to(device)
        new_output = model(new_data)
        reg_loss += criterion(new_output, output)

    return reg_loss