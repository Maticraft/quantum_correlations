import torch
import torch.nn as nn

from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector

from commons.pytorch_utils import CNN


class HybridQCNN(CNN):
    def __init__(self, qbits_num, cnn_output_size, output_size, device, conv_num = 3, fc_num = 5, kernel_size = 2, filters_ratio = 16, dilation = 1, ratio_type = 'sqrt'):
        super(HybridQCNN, self).__init__(qbits_num, cnn_output_size, conv_num, fc_num, kernel_size, filters_ratio, dilation, ratio_type)
        self.simulator = Aer.get_backend('aer_simulator')
        self.device = device
        self.qi = QuantumInstance(self.simulator)
        self.qnn = TorchConnector(self.make_qnn(cnn_output_size, cnn_output_size, self.qi))
        self.post_process = nn.Linear(cnn_output_size, output_size)
    
    def make_qnn(self, input_size, output_size, qi):
        feature_map = ZZFeatureMap(input_size)
        ansatz = RealAmplitudes(input_size, reps=1)
        qc = QuantumCircuit(input_size)
        qc.append(feature_map, range(input_size))
        qc.append(ansatz, range(input_size))

        # Define CircuitQNN and initial setup
        parity = lambda x: "{:b}".format(x).count("1") % 2  # optional interpret function
        qnn = CircuitQNN(
            qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=parity,
            output_shape=output_size,
            quantum_instance=qi,
            input_gradients=True
        )     

        return qnn

    def forward(self, x):
        cnn_out = super().forward(x) #.cpu()
        qnn_out = self.qnn(cnn_out).double() #.to(self.device)
        output = self.post_process(qnn_out)
        output = torch.sigmoid(output)
        return output
