#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from commons.models.cnns import CNN
from commons.models.siamese_networks import VectorSiamese
from commons.metrics import generate_parametrized_qs, global_entanglement_bipartitions
from commons.models.separators import FancySeparator
from commons.pytorch_utils import loc_op_circ, loc_op, all_perms

import torch
from commons.pytorch_utils import separator_predict

# Gif params
gif_pngs = 100
thresh_bound = (0, 0.02)

qbits_num = 3

# Separator params
thresh = 0.0165
out_channels_per_ratio = 24
input_channels = 2
criterion = 'L1'

sep_save_path = './classifiers/FancySeparator_l1_pure_sep_3q_o48.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = FancySeparator(qbits_num, out_channels_per_ratio, input_channels)
model.load_state_dict(torch.load(sep_save_path))
model.double()
model.to(device)
#--------------------------------------------------------------------------------------------

# Data params
a = 1/np.sqrt(2)
cs = np.arange(0, 1+ 0.05, 0.05)
fis = np.append(np.arange(0*np.pi - 0.4, 2*np.pi + 0.4, 0.1), 2*np.pi)
qubits_num = 3

#--------------------------------------------------------------------------------------------
step = (thresh_bound[1] - thresh_bound[0]) / gif_pngs
thresholds = np.arange(thresh_bound[0], thresh_bound[1] + step, step)


for gif_num, thresh in enumerate(thresholds):
    print('Current threshold = {}'.format(round(thresh, 4)))
    # Plot
    res_save_path = "./plots/gif/a=b/discord_map3q_pred_sep_{}.png".format(gif_num)


    fig = plt.figure()
    plt.title("Threshold = {}".format(thresh))

    for c in cs:
        for fi in fis:
            rho = generate_parametrized_qs(qubits_num, a, c, fi)

            torch_rho = torch.unsqueeze(torch.tensor(rho.data), dim=0)
            torch_data = torch.stack((torch_rho.real, torch_rho.imag), dim= 1)

            torch_data = loc_op_circ(torch_data).double()    
            prediction = separator_predict(model, device, torch_data, thresh, criterion).item()

            if prediction > 0.5:
                color = 'blue'
            else:
                color = 'red'
                
            plt.scatter(fi, c, c=color)

    red_patch = mpatches.Patch(color='red', label='predicted 1')
    blue_patch = mpatches.Patch(color='blue', label='predicted 0')
    lgd = fig.legend(handles=[red_patch, blue_patch], bbox_to_anchor = (0.96, 0.95))
    
    plt.xlabel(r"$\phi$")
    plt.ylabel("c")
    plt.tight_layout(pad = 2)

    plt.savefig(res_save_path)
    plt.close()