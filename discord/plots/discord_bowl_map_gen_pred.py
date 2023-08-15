#%%
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from commons.metrics import generate_parametrized_qs, global_entanglement_bipartitions
from math import tan
from commons.models.separators import FancySeparator
from commons.pytorch_utils import loc_op_circ, all_perms
import torch

from commons.test.separator import separator_predict
from commons.trace import trace_predict

plt.rc('font', size=16) #controls default text size
plt.rc('axes', titlesize=18) #fontsize of the title


# General params
thresh = 0.001
map_mode = 'sep_loss' # possible modes: 'sep_loss', 'sep_pred', 'trace_loss', 'trace_pred'
log_scale = False
entanglement_type = '3q'
randomize_amplitudes = False
criterion = 'L1'
qubits_num = 3


# Map params
bowl_params = {
    'c': 0.7, #decoherence
    'c_shift': 0.005, #0.005
    'a': 1/np.sqrt(2), #amplitude
    'a_shift': 0.005, #0.005
    'phi': 0., #phase
    'walls': 'phi', #axis for walls height alignement
    'floor_rad': 8, #radius of the floor in the units of shift params
    'angle': 0.895*np.pi/2, #angle between walls and floor (0 - horizontal alignement, pi/2 - vertical) 
    'angle_shift': 0.0*np.pi/2. # shift of the angle (if equal to 0 const angle will be applied)
}


#----------------------------------------------------------------------
# Separator params
out_channels_per_ratio = 24
input_channels = 2
fc_layers = 4

#sep_save_path = './classifiers/FancySeparator_l1_pure_sep_3q_o48.pt'
#sep_save_path = './models/FancySeparator_l1_pure_sep_param_3q_o48_bl.pt'
#sep_save_path = './classifiers/FancySeparatorSymTrue_l1_sep_3q_o48_bl.pt'
#sep_save_path = './classifiers/FancySeparator_l1_pure_sep_param_3q_o2_bl.pt'
#sep_save_path = './classifiers/FancySeparatorSymTrue_l1_sep_sym_3q_o48_bl.pt'
#sep_save_path = './classifiers/SiamFancySeparatorSymTrue_l1_sep_sym_3q_o48_bl.pt'
#sep_save_path = './classifiers/SiamFancySeparatorOutSym_l1_sep_sym_3q_o48_bl.pt'

#sep_save_path = './classifiers/FancySeparator_l1_npzd_3q_o48_bl.pt'
#sep_save_path = './classifiers/FancySeparator_l1_npzd_3q_o48_fc4_bl.pt'
#sep_save_path = './classifiers/FancySeparator_l1_sep_3q_o48_fc4_bl.pt'
sep_save_path = './classifiers/FancySeparator_l1_nps_3q_o48_fc4_bl.pt'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

model = FancySeparator(qubits_num, out_channels_per_ratio, input_channels, fc_layers)
model.load_state_dict(torch.load(sep_save_path))
model.double()
model.to(device)
#----------------------------------------------------------------------
# Functions

def shift_param(param, shift, old_value):
    if param == 'a':
        return np.sqrt(old_value ** 2 + shift)
    else:
        return old_value + shift


def check_bounds(param, val):
    if param == 'phi':
        if val >= - 0.1 and val <= 2*np.pi + 0.1:
            return True
        else:
            return False
    if param == 'a':
        if val >= 0.1 and val <= 0.9:
            return True
        else:
            return False
    else:
        if val >= 0. and val <= 1.:
            return True
        else:
            return False

def display_str(param):
    if param != 'phi':
        return param
    else:
        return r'$\phi$'



def find_closest_val(x, y, map):
    try:
        val = map[(round(x, 2), round(y, 2))]
    except:
        min_dist = 100
        val = list(map.values())[0]
        for x_i, y_i in map.keys():
            x_dist = abs(x - x_i)
            y_dist = abs(y - y_i)
            dist = x_dist + y_dist
            if dist < min_dist:
                min_dist = dist
                val = map[(x_i, y_i)]
         
    return val

def next_pos(pos, idx, k):
    n_pos = pos.copy()

    if k % 2 == 0:    
        if k / 2 > bowl_params['floor_rad']:
            n_pos[z] = shift_param(z, tan(bowl_params['angle']) * bowl_params[floor_axis[idx] + '_shift'], n_pos[z])

            if bowl_params['angle'] < 0.95*np.pi/2:
                bowl_params['angle'] += bowl_params['angle_shift']
            
            if bowl_params['angle'] > 0.95*np.pi/2:
                bowl_params['angle'] = 0.95*np.pi/2
            
        n_pos[floor_axis[idx]] = shift_param(floor_axis[idx], bowl_params[floor_axis[idx] + '_shift'], n_pos[floor_axis[idx]])

    n_pos[floor_axis[idx]] = 2 * bowl_params[floor_axis[idx]] - n_pos[floor_axis[idx]]
  
    return n_pos


# Execution 

if log_scale:
    norm = mcolors.LogNorm(vmin=0.0005, vmax=0.02)
else:
    norm = mcolors.Normalize(vmin=0., vmax=0.02)

cmap = plt.cm.jet.reversed()
cmap = cmap(np.arange(256))
cmap = np.vstack((np.tile(cmap[0],(7,1)),cmap[:210]))
cmap = mcolors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])

print('Current threshold = {}'.format(round(thresh, 4)))

fig, ax = plt.subplots(figsize=(6, 8))

if map_mode == 'pred':
    ax.set_title("Threshold = {}".format(round(thresh, 4)))

z = bowl_params['walls']
angle_0 = bowl_params['angle']
floor_axis = [k for k in ['a','c','phi'] if k != z]

pos = {
        floor_axis[0]: bowl_params[floor_axis[0]],
        floor_axis[1]: bowl_params[floor_axis[1]],
        z: bowl_params[z]  
    }

min_x, max_x = 1., 0
min_y, max_y = 1., 0.
xy_to_z = {}

i = 0
while check_bounds(floor_axis[0], pos[floor_axis[0]]):
    if pos[floor_axis[0]] < min_x:
        min_x = pos[floor_axis[0]]
    if pos[floor_axis[0]] > max_x:
        max_x = pos[floor_axis[0]]

    pos[floor_axis[1]] = bowl_params[floor_axis[1]]
    pos_zi = pos[z]
    angle = bowl_params['angle']

    j = 0
    while check_bounds(floor_axis[1], pos[floor_axis[1]]) and check_bounds(z, pos[z]):
        if pos[floor_axis[1]] > max_y:
            max_y = pos[floor_axis[1]]
        if pos[floor_axis[1]] < min_y:
            min_y = pos[floor_axis[1]]

        xy_to_z[(round(pos[floor_axis[0]], 2), round(pos[floor_axis[1]], 2))] = pos[z]

        if entanglement_type == '3q':
            rho = generate_parametrized_qs(qubits_num, pos['a'], pos['c'], 0, pos['phi'], random_amplitude_swap=randomize_amplitudes)
        elif entanglement_type == '2q':
            rho = generate_parametrized_qs(qubits_num, pos['a'], pos['c'], pos['phi'], random_amplitude_swap=randomize_amplitudes)
        else:
            raise ValueError('Wrong entanglement type')

        torch_rho = torch.unsqueeze(torch.tensor(rho.data), dim=0)
        torch_data = torch.stack((torch_rho.real, torch_rho.imag), dim= 1)

        torch_data = loc_op_circ(torch_data).double()
        # perms = all_perms(torch_data)
        # perm_indx = np.random.randint(0, len(perms))
        # torch_data = perms[perm_indx].double()

        if map_mode == 'sep_pred':  
            prediction = separator_predict(model, device, torch_data, thresh, criterion).item()
            if prediction >= 0.5:
                color = 'blue'
            else:
                color = 'red'
        elif map_mode == 'sep_loss':
            prediction, loss = separator_predict(model, device, torch_data, thresh, criterion, return_loss=True)
            prediction, loss = prediction.item(), loss.item()
            color = cmap(norm(loss))
        elif map_mode == 'trace_pred':  
            prediction = trace_predict(torch_data, thresh, criterion).item()
            if prediction >= 0.5:
                color = 'blue'
            else:
                color = 'red'
        elif map_mode == 'trace_loss':
            prediction, loss = trace_predict(torch_data, thresh, criterion, return_measure_value=True)
            prediction, loss = prediction.item(), loss.item()
            color = cmap(norm(loss))

        neg = global_entanglement_bipartitions(rho, "negativity")
        disc = global_entanglement_bipartitions(rho, "discord")
        
        
        old_angle = bowl_params['angle']
        n_pos_x = next_pos(next_pos(pos, 0, i), 0, i + 1)
        bowl_params['angle'] = old_angle
        n_pos_y = next_pos(next_pos(pos, 1, j), 1, j + 1)
        bowl_params['angle'] = old_angle

        if neg < 0.001:    
            if check_bounds(floor_axis[0], n_pos_x[floor_axis[0]]):
                if entanglement_type == '3q':
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_x['a'], n_pos_x['c'], 0, n_pos_x['phi'])
                else:
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_x['a'], n_pos_x['c'], n_pos_x['phi'])
                next_neg = global_entanglement_bipartitions(next_rho, "negativity")
                if next_neg > 0.001:
                    color = 'black'

            if check_bounds(floor_axis[1], n_pos_y[floor_axis[1]]):
                if entanglement_type == '3q':
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_y['a'], n_pos_y['c'], 0, n_pos_y['phi'])
                else:
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_y['a'], n_pos_y['c'], n_pos_y['phi'])
                next_neg = global_entanglement_bipartitions(next_rho, "negativity")
                if next_neg > 0.001:
                    color = 'black'
        else:
            if check_bounds(floor_axis[0], n_pos_x[floor_axis[0]]):
                if entanglement_type == '3q':
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_x['a'], n_pos_x['c'], 0, n_pos_x['phi'])
                else:
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_x['a'], n_pos_x['c'], n_pos_x['phi'])
                next_neg = global_entanglement_bipartitions(next_rho, "negativity")
                if next_neg <= 0.001:
                    color = 'black'

            if check_bounds(floor_axis[1], n_pos_y[floor_axis[1]]):
                if entanglement_type == '3q':
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_y['a'], n_pos_y['c'], 0, n_pos_y['phi'])
                else:
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_y['a'], n_pos_y['c'], n_pos_y['phi'])
                next_neg = global_entanglement_bipartitions(next_rho, "negativity")
                if next_neg <= 0.001:
                    color = 'black'
            
        if disc < 0.001:
            if check_bounds(floor_axis[0], n_pos_x[floor_axis[0]]):
                if entanglement_type == '3q':
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_x['a'], n_pos_x['c'], 0, n_pos_x['phi'])
                else:
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_x['a'], n_pos_x['c'], n_pos_x['phi'])
                next_disc = global_entanglement_bipartitions(next_rho, "discord")
                if next_disc > 0.001:
                    color = 'green'

            if check_bounds(floor_axis[1], n_pos_y[floor_axis[1]]):
                if entanglement_type == '3q':
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_y['a'], n_pos_y['c'], 0, n_pos_y['phi'])
                else:
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_y['a'], n_pos_y['c'], n_pos_y['phi'])
                next_disc = global_entanglement_bipartitions(next_rho, "discord")
                if next_disc > 0.001:
                    color = 'green'


        ax.scatter(pos[floor_axis[0]], pos[floor_axis[1]], color=color, marker = 's', s=5)

        pos = next_pos(pos, 1, j)
        j += 1

    pos[z] = pos_zi
    bowl_params['angle'] = angle
    pos = next_pos(pos, 0, i)
    i += 1

bowl_params['angle'] = angle_0   

red_patch = mpatches.Patch(color='red', label='predicted 1')
blue_patch = mpatches.Patch(color='blue', label='predicted 0')
green_patch = mpatches.Patch(color='green', label='zero discord boundary')
black_patch = mpatches.Patch(color='black', label='separable states boundary')
#lgd = fig.legend(handles=[red_patch, blue_patch, green_patch, black_patch], bbox_to_anchor = (0.9, 0.9))

# Split the axes
ax_r = ax.secondary_yaxis('right')
ax_t = ax.secondary_xaxis('top')

x_shift = bowl_params[floor_axis[0] + '_shift'] * bowl_params['floor_rad']
#xs = np.arange(min_x, max_x + x_shift, x_shift)
xs = np.linspace(min_x, max_x, 11)
a_ticks = [str(round(x, 2)) for x in xs]
x_phi_ticks = [str(round(find_closest_val(x, bowl_params[floor_axis[1]], xy_to_z), 2)) for x in xs]
ax.set_xticks(xs, labels=a_ticks)
ax.xaxis.set_tick_params(rotation=90)
ax_t.set_xticks(xs, labels=x_phi_ticks)
ax_t.xaxis.set_tick_params(rotation=90)


y_shift = 1.0*bowl_params[floor_axis[1] + '_shift'] * bowl_params['floor_rad']
#ys = np.arange(min_y, max_y + y_shift, y_shift)
ys = np.linspace(min_y, max_y, 11)
c_ticks = [str(round(y, 2)) for y in ys]
y_phi_ticks = [str(round(find_closest_val(bowl_params[floor_axis[0]], y, xy_to_z), 2)) for y in ys]
ax.set_yticks(ys, labels=c_ticks)
ax_r.set_yticks(ys, labels=y_phi_ticks)

ax.set_xlabel(display_str(floor_axis[0]))
ax_t.set_xlabel(display_str(z))
ax.set_ylabel(display_str(floor_axis[1]))
ax_r.set_ylabel(display_str(z))


if map_mode == 'sep_loss' or map_mode == 'trace_loss':
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size='5%', pad=1.1, pack_start = True)
    fig.add_axes(cax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='loss', cax=cax, orientation='horizontal')


#plt.savefig("./plots/discord_bowl_map3q_pred_sep_th{}.png".format(thresh), bbox_extra_artists=[lgd], pad_inches = 0.2, bbox_inches='tight')
if log_scale:
    plt.savefig("./plots/discord_bowl_map3q_{}_ent{}_{}_th{}_log.png".format(map_mode, entanglement_type, criterion, thresh), pad_inches = 0.2, bbox_inches='tight')
else:
    plt.savefig("./plots/discord_bowl_map3q_{}_ent{}_{}_th{}_npsepfc4.png".format(map_mode, entanglement_type, criterion, thresh), pad_inches = 0.2, bbox_inches='tight')

plt.close()

#%%
