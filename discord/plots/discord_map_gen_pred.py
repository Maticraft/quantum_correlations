#%%
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from commons.models.cnns import CNN
from commons.models.siamese_networks import VectorSiamese
from commons.metrics import generate_parametrized_qs, global_entanglement_bipartitions
from commons.models.separators import FancySeparator
from commons.pytorch_utils import loc_op_circ, loc_op, all_perms

from tqdm import tqdm
import torch
from commons.pytorch_utils import separator_predict

model_type = 'Separator'
map_mode = 'loss'

# Model params + loading
qbits_num = 3
output_size = 3
dilation = 1
cn = 3
fn = 5
kernel_size = 2
fr = 16
ch = 32
cb = 1
ratio_type = 'sqrt'
pooling = 'None'
mode = 'classifier'
batch_norm = True

# Additional params for Siamese model
biparts_mode = 'all'

# Separator params
thresh = 0.0165
out_channels_per_ratio = 24
input_channels = 2
criterion = 'L1'


res_save_path = "./plots/discord_map3q_pred_sep_loss.png"

siam_save_path = './classifiers/3qubits_disc/vectorClassifier_3q_siam_pure_bal_param_clean_cn_3_k2_ep50.pt'

save_path = './classifiers/3qubits_disc/vectorClassifier_3q_pure_bal_cn3_k2_ep6.pt'
#save_path = './classifiers/3qubits_disc/vectorClassifier_3q_pure_bal_param_cn3_k2_ep20.pt'

sep_save_path = './classifiers/FancySeparator_l1_pure_sep_3q_o48.pt'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('Using device:', device)


if model_type == 'CNN':
    model = CNN(qbits_num, output_size, cn, fn, kernel_size, fr, dilation = dilation, ratio_type= ratio_type, pooling = pooling, mode = mode, cn_in_block=cb, batch_norm=batch_norm)
    model.double()
    model.load_state_dict(torch.load(save_path))
    model.to(device)

elif model_type == 'Siamese':
    model = VectorSiamese(qbits_num, output_size, cn, fn, kernel_size, fr, dilation = dilation, ratio_type= ratio_type, pooling = pooling, mode = mode, biparts_mode = biparts_mode, tensor_layers=False, tensor_map=False, batch_norm = batch_norm)
    model.load_state_dict(torch.load(siam_save_path))   
    model.double()
    model.to(device)

elif model_type == 'Separator':
    model = FancySeparator(qbits_num, out_channels_per_ratio, input_channels)
    model.load_state_dict(torch.load(sep_save_path))
    model.double()
    model.to(device)
#--------------------------------------------------------------------------------------------

# Data params
a = 1/np.sqrt(2)
c_step = 0.005
cs = np.arange(-0.05, 1 + 0.05, c_step)
fi_step = 0.05
fis = np.arange(0*np.pi - 0.4, 2*np.pi, fi_step)
fis = np.concatenate((fis, np.arange(2*np.pi, 2*np.pi + 0.4, fi_step)))
fis = np.sort(fis)
qubits_num = 3

#--------------------------------------------------------------------------------------------

# Plot

cmap = plt.cm.jet.reversed()
cmap = cmap(np.arange(256))
cmap = np.vstack((np.tile(cmap[0],(7,1)),cmap[:210]))
cmap = mcolors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

norm = mcolors.Normalize(vmin=0.0, vmax=0.02)


fig, ax = plt.subplots(figsize=(6, 8))

for c in tqdm(cs, desc='Plot generation'):
    for i, fi in enumerate(fis):
        rho = generate_parametrized_qs(qubits_num, a, c, fi)

        torch_rho = torch.unsqueeze(torch.tensor(rho.data), dim=0)
        torch_data = torch.stack((torch_rho.real, torch_rho.imag), dim= 1)

        torch_data = loc_op_circ(torch_data).double()    
        if model_type == 'Siamese':
            torch_data = torch.unsqueeze(torch_data, dim = 0)

        if model_type != 'Separator':
            prediction = torch.round(torch.mean(model(torch_data.to(device)))).item()
        else:
            prediction, loss = separator_predict(model, device, torch_data, thresh, criterion, return_loss=True)
            prediction, loss = prediction.item(), loss.item()
            if map_mode == 'loss':
                color = cmap(norm(loss))
            else:
                if prediction > 0.5:
                    color = 'blue'
                else:
                    color = 'red'

        neg = global_entanglement_bipartitions(rho, "negativity")
        disc = global_entanglement_bipartitions(rho, "discord")

        if i < len(fis) - 1:
            next_rho_c = generate_parametrized_qs(qubits_num, a, c + c_step, fi)
            next_rho_fi = generate_parametrized_qs(qubits_num, a, c, fis[i+1])
            next_neg_c = global_entanglement_bipartitions(next_rho_c, "negativity")
            next_neg_fi = global_entanglement_bipartitions(next_rho_fi, "negativity")
            next_disc_c = global_entanglement_bipartitions(next_rho_c, "discord")
            next_disc_fi = global_entanglement_bipartitions(next_rho_fi, "discord")

            if neg < 0.001: 
                if next_neg_fi > 0.001 or next_neg_c > 0.001:   
                    color = 'black'
            else:
                if next_neg_fi <= 0.001 or next_neg_c <= 0.001:
                    color = 'black'
                
            if disc < 0.001:
                if next_disc_fi > 0.001 or next_disc_c > 0.001:
                    color = 'green'
            else:
                if next_disc_fi <= 0.001 or next_disc_c <= 0.001:
                    color = 'green'

        if i > 0:
            prev_rho_c = generate_parametrized_qs(qubits_num, a, c - c_step, fi)
            prev_rho_fi = generate_parametrized_qs(qubits_num, a, c, fis[i-1])
            prev_neg_c = global_entanglement_bipartitions(prev_rho_c, "negativity")
            prev_neg_fi = global_entanglement_bipartitions(prev_rho_fi, "negativity")
            prev_disc_c = global_entanglement_bipartitions(prev_rho_c, "discord")
            prev_disc_fi = global_entanglement_bipartitions(prev_rho_fi, "discord")

            if neg < 0.001: 
                if prev_neg_fi > 0.001 or prev_neg_c > 0.001:   
                    color = 'black'
            else:
                if prev_neg_fi <= 0.001 or prev_neg_c <= 0.001:
                    color = 'black'
                
            if disc < 0.001:
                if prev_disc_fi > 0.001 or prev_disc_c > 0.001:
                    color = 'green'
            else:
                if prev_disc_fi <= 0.001 or prev_disc_c <= 0.001:
                    color = 'green'

        if i > 0 and i < len(fis) - 1:
            if (prev_disc_fi <= 0.001 and next_disc_fi <= 0.001) or (prev_disc_c <= 0.001 and next_disc_c <= 0.001):
                color = 'green'

            
        ax.scatter(fi, c, color=color, s=5)

  
ax.set_xlabel(r"$\phi$")
ax.set_ylabel("c")

if map_mode == 'loss':
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size='5%', pad=0.7, pack_start = True)
    fig.add_axes(cax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=cax, label='loss', orientation='horizontal', ticks=[0, 0.005, 0.01, 0.015, 0.02])
else:
    red_patch = mpatches.Patch(color='red', label='predicted 1')
    blue_patch = mpatches.Patch(color='blue', label='predicted 0')
    lgd = fig.legend(handles=[red_patch, blue_patch], bbox_to_anchor = (0.96, 0.95))

plt.savefig(res_save_path,  pad_inches = 0.2, bbox_inches='tight')