import torch
import torch.nn as nn

class Ensemble(nn.Module):
    def __init__(self, ensemble_submodel_class, ensemble_submodel_params, ensemble_size, selector_class, selector_params):
        super(Ensemble, self).__init__()
        self.ensemble_size = ensemble_size
        self.submodels = nn.ModuleList([ensemble_submodel_class(**ensemble_submodel_params) for _ in range(ensemble_size)])
        self.selector = nn.Sequential(
            selector_class(**selector_params),
            nn.Softmax(dim = 1)
        )

    def forward(self, x, return_all = False):
        submodels_weights = self.selector(x) # (batch_size, ensemble_size)
        submodels_outputs = torch.stack([submodel(x) for submodel in self.submodels], dim=1) # (batch_size, ensemble_size, output_size)
        if return_all:
            return submodels_outputs, submodels_weights
        best_models_ids = torch.argmax(submodels_weights, dim = 1, keepdim = True).unsqueeze(-1).expand(-1, -1, submodels_outputs.shape[-1])
        return torch.gather(submodels_outputs, dim=1, index=best_models_ids).squeeze(1) # (batch_size, output_size)
    