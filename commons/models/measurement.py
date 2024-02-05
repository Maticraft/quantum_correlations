import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, layers: int, input_dim: int, hidden_size: int, output_dim: int):
        super(MLP, self).__init__()
        mlp = []
        mlp.append(nn.Linear(input_dim, hidden_size))
        mlp.append(nn.ReLU())
        for i in range(layers - 1):
            mlp.append(nn.Linear(hidden_size, hidden_size))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(hidden_size, output_dim))
        self.mlp = nn.Sequential(*mlp)
    
    def forward(self, x: torch.Tensor):
        return self.mlp(x)
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class DensityMatrixReconstructor(nn.Module):
    def __init__(self, input_dim: int, num_qubits: int, layers: int = 2, hidden_size: int = 16):
        super(DensityMatrixReconstructor, self).__init__()
        self.num_qubits = num_qubits
        self.mlp = MLP(layers, input_dim, hidden_size, 2 * (4 ** num_qubits))

    def forward(self, measurements: torch.Tensor):
        return self.mlp(measurements).view(-1, 2, 2 ** self.num_qubits, 2 ** self.num_qubits)
    

class Classifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layers: int = 2, hidden_size: int = 16):
        super(Classifier, self).__init__()
        self.mlp = MLP(layers, input_dim, hidden_size, output_dim)

    def forward(self, x: torch.Tensor):
        out = self.mlp(x)
        return torch.sigmoid(out)
    