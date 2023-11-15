import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
from commons.models.separators import FancySeparator

from commons.trace import trace_predict
from commons.metrics import generate_parametrized_np_qs, global_entanglement_bipartitions, local_randomize_matrix
from commons.test_utils.separator import separator_predict


qubits_num = 3
trace_thresh = 1.e-5
map_mode = 'sep_loss' # possible modes: 'sep_loss', 'trace_loss', 'real_metrics'
plot_mode = 'contourf' # possible modes: 'scatter', 'contourf'
criterion = 'bures'
log_scale = True
min_loss = 1.e-3
max_loss = 0.1
resolution = 100

# Params
a_max = 1/np.sqrt(2)
a_min = 0

p_max = 1/2
p_min = 0

c_max = 1
c_min = 0

fi_max = np.pi/2
fi_min = 0

# NN params
sep_save_path = './paper_models/3qbits/FancySeparator_l1_all_sep_o48_fc4_bl.pt'
out_channels_per_ratio = 24
input_channels = 2
fc_layers = 4


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

model = FancySeparator(qubits_num, out_channels_per_ratio, input_channels, fc_layers)
model.load_state_dict(torch.load(sep_save_path))
model.double()
model.to(device)

# regional borders
x_min = 0 # Fixed, won't work if changed
x_01 = 3/8
x_12 = 5/8
x_max = 1 # Fixed, won't work if changed
y_min = 0 # Fixed, won't work if changed
y_01 = 3/8
y_12 = 5/8
y_max = 1 # Fixed, won't work if changed

#----------------------------------------------------------------------
# Functions

def get_params(x, y):
    assert x >= x_min and x <= x_max and y >= y_min and y <= y_max, "wrong x or y"
    if y >= 1 - x:
        a, fi, p, c = get_std_params(x, y)
    else:
        a, fi, p, c = get_std_params(1 - y, 1 - x)
        c += (abs(x + y - 1)/np.sqrt(2)) * (c_max - c)
    return a, fi, p, c

def get_std_params(x, y):
    if y > y_12:
        a = a_max
        fi = (y - y_12)/(y_max - y_12) * (fi_max - fi_min) + fi_min
        if x < x_01:
            p = p_min
            c = ((abs(x - y)/np.sqrt(2)) - (abs(x_01 - y_12)/np.sqrt(2))) * (c_max - c_min) + c_min
        elif x < x_12:
            p = (x - x_01)/(x_12 - x_01) * (p_max - p_min) + p_min
            c = c_min
        else:
            p = p_max
            c = ((abs(x + y - 1)/np.sqrt(2)) - (abs(x_12 + y_12 - 1)/np.sqrt(2))) * (c_max - c_min) + c_min
    elif y > y_01:
        c = c_min
        a = (y - y_01)/(y_12 - y_01) * (a_max - a_min) + a_min
        if x < x_01:
            raise ValueError("Something went wrong")
        elif x < x_12:
            fi = fi_min
            p = (x - x_01)/(x_12 - x_01) * (p_max - p_min) + p_min
        else:
            p = p_max
            fi = (x - x_12)/(x_max - x_12) * (fi_max - fi_min) + fi_min
    else:
        if x < x_12:
            raise ValueError("Something went wrong")
        else:
            a = a_min
            p = p_max
            fi = (x - x_12)/(x_max - x_12) * (fi_max - fi_min) + fi_min
            c = ((abs(x - y)/np.sqrt(2)) - (abs(x_12 - y_01)/np.sqrt(2))) * (c_max - c_min) + c_min
    return a, fi, p, c

def get_metrics(x, y, map_mode):
    a, fi, p, c = get_params(x, y)
    rho = generate_parametrized_np_qs(a, p, fi, c, qubits_num)
    torch_data = torch.unsqueeze(torch.tensor(rho.data), dim=0)
    torch_data = torch.stack((torch_data.real, torch_data.imag), dim=1)
    torch_data = torch_data.to(device)

    if map_mode == 'sep_loss':
        prediction, loss = separator_predict(model, device, torch_data, 1.e-3, criterion, return_loss=True)
        prediction, loss = prediction.item(), loss.item()
        return loss
    elif map_mode == 'trace_loss':
        prediction, loss = trace_predict(torch_data, trace_thresh, criterion, return_measure_value=True)
        prediction, loss = prediction.item(), loss.item()
        return loss
    elif map_mode == 'real_metrics':
        neg = global_entanglement_bipartitions(rho, "negativity")
        disc = global_entanglement_bipartitions(rho, "discord")
        prediction, loss = trace_predict(torch_data, trace_thresh, criterion, return_measure_value=True)
        prediction, loss = prediction.item(), loss.item()
        if prediction == 0:
            return min_loss
        elif disc < 0.001:
            return (max_loss - min_loss) / 3
        elif neg < 0.001:
            return 2*(max_loss - min_loss) / 3
        else:
            return max_loss
    else:
        raise ValueError("Wrong map_mode")


#----------------------------------------------------------------------
# Execution

if log_scale:
    norm = mcolors.LogNorm(vmin=min_loss, vmax=max_loss)
else:
    norm = mcolors.Normalize(vmin=min_loss, vmax=max_loss)

cmap = plt.cm.jet.reversed()
cmap = cmap(np.arange(256))
cmap = np.vstack((np.tile(cmap[0],(7,1)),cmap[:210]))
cmap = mcolors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

xs = np.linspace(x_min, x_max, resolution)
ys = np.linspace(y_min, y_max, resolution)
zs = [[get_metrics(x, y, map_mode) for x in xs] for y in ys]
zs = np.array(zs)
res_x = 0
res_y = 0
fig, ax = plt.subplots(figsize = (6, 6))
if map_mode == 'real_metrics':
    levels = [min_loss, (max_loss - min_loss) / 3, 2*(max_loss - min_loss) / 3, max_loss]
    cs = ax.contourf(xs, ys, zs, 3, hatches=['/', '\\', '//', '\\\\'],
                   colors=('lightcyan', 'skyblue', 'royalblue', 'navy'))
else: 
    if plot_mode == 'scatter':
        res_x = (x_max - x_min) / (2 * resolution)
        res_y = (y_max - y_min) / (2 * resolution)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                color = cmap(norm(zs[j, i]))
                ax.scatter(x, y, color=color)
    elif plot_mode == 'contourf':
        cs = ax.contourf(xs, ys, zs, resolution, cmap=cmap, norm=norm, extend='both', alpha=1)


# Plot lines for axis
# Plot horizontal lines
ax.plot([x_min, x_max], [y_01 - res_y, y_01 - res_y], color='black', linestyle='--')
ax.plot([x_min, x_max], [y_12 - res_y, y_12 - res_y], color='black', linestyle='--')
# Plot vertical lines
ax.plot([x_01 - res_x, x_01 - res_x], [y_min, y_max], color='black', linestyle='--')
ax.plot([x_12 - res_x, x_12 - res_x], [y_min, y_max], color='black', linestyle='--')

ax.axis('off')

if map_mode == 'sep_loss' or map_mode == 'trace_loss':
    if map_mode == 'sep_loss':
        plt.title('Separator loss map')
    elif map_mode == 'trace_loss':
        plt.title('Trace loss map')
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size='5%', pad=0.5, pack_start = True)
    fig.add_axes(cax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if log_scale:
        plt.colorbar(sm, label='loss', cax=cax, orientation='horizontal')
    else:  
        plt.colorbar(sm, label='loss', cax=cax, orientation='horizontal', ticks=[min_loss, (max_loss + min_loss) / 2, max_loss])

    plt.tight_layout(pad = 0.7)

    if log_scale:
        plt.savefig("./plots/discord_{}_fancy_bowl_asym_map{}q_{}_loss_{}_log.png".format(map_mode, qubits_num, criterion, plot_mode)) #, bbox_extra_artists=[lgd], bbox_inches='tight')
    else:
        plt.savefig("./plots/discord_{}_fancy_bowl_asym_map{}q_{}_loss_{}.png".format(map_mode, qubits_num, criterion, plot_mode)) #, bbox_extra_artists=[lgd], bbox_inches='tight')
else:
    plt.title('Analytical map')

    artists, labels = cs.legend_elements(str_format='{:2.4f}'.format)
    labels = ['product', 'non-product\nzero-discord', 'discordant\nseparable', 'entangled']
    lgd = fig.legend(artists, labels, bbox_to_anchor = (0.96, 0.95))

    plt.tight_layout(pad = 0.7, rect=[0., 0.4, 0.7, 1])

    plt.savefig("./plots/discord_fancy_bowl_asym_map{}q.png".format(qubits_num), bbox_extra_artists=[lgd], bbox_inches='tight')