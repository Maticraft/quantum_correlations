import numpy as np
import torch

from commons.data.pure_states_generator import generate_pure_test, generate_mixed_test_def
from commons.data.mixed_reduced_states_generator import generate_mixed_reduced_test
from commons.models.separators import FancySeparator
from commons.pytorch_utils import separator_predict

num_qubits = 3
data_path = f'./data/{num_qubits}qubits/'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = './models/FancySeparator_l1_pure_sep_param_3q_o48_bl.pt'
model = FancySeparator(num_qubits, 48, 2)
model.load_state_dict(torch.load(model_path))
model.double()
model.to(device)

# Generate train data set
data_type = 'train'
indx = 0
for i in range(3):
    print("Iteration: ", i)
    indx = generate_pure_test(num_qubits, encoded=False, indx=indx, data_type=data_type, discord=True)
    indx = generate_mixed_test_def(num_qubits, encoded=False, indx=indx, data_type=data_type, discord=True)
    indx = generate_mixed_reduced_test(num_qubits, encoded=False, indx=indx, data_type=data_type, discord=True)

with open(data_path + f'{data_type}_separator_loss.txt', 'w') as f:
    for i in range(indx):
        rho = np.load(data_path + data_type + f'matrices/dens{i}.npy')
        matrix_r = np.real(rho)
        matrix_im = np.imag(rho)

        tensor = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0))
        pred, loss = separator_predict(model, device, tensor, 0.1, return_loss=True)
        f.write(f'dens{i}.npy, {loss}\n')

