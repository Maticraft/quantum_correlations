import torch
import torch.nn as nn


class EigvalsClassifier(nn.Module):
    def __init__(self, qbits_num, output_size, input_channels = 2, fc_num = 4, linear_transforms = 1, hidden_size = 128):
        super(EigvalsClassifier, self).__init__()
        self.dim = 2**qbits_num
        self.output_size = output_size
        self.fc_num = fc_num
        self.input_channels = input_channels
        self.hidden_size = hidden_size

        self.linear_transforms = nn.ModuleList([nn.Linear(self.dim*self.dim, self.dim*self.dim) for _ in range(linear_transforms)])
        self.fc_layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(self.dim*(linear_transforms+1), self.hidden_size),
                nn.ReLU()
            ),
            *[
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU()
                ) for _ in range(fc_num - 1)
            ],
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_transform = x.view(-1, self.input_channels, self.dim*self.dim)
        x_transform = torch.stack([
            torch.stack((linear_transform(x_transform[:, 0]), linear_transform(x_transform[:, 1])), dim = 1)
            for linear_transform in self.linear_transforms
        ], dim = 1)
        x_t_complex = torch.complex(x_transform[:, :, 0], x_transform[:, :, 1]).view(-1, x_transform.shape[1], self.dim, self.dim)
        x_complex = torch.complex(x[:, 0], x[:, 1]).unsqueeze(1)
        x_cat = torch.cat((x_complex, x_t_complex), dim = 1)

        eigvals = torch.linalg.eigvals(x_cat).real.view(-1, x_cat.shape[1]*x_cat.shape[2])
        output = self.fc_layers(eigvals)
        return output
