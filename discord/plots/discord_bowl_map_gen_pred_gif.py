from re import X
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from commons.models.cnns import CNN
from commons.models.siamese_networks import VectorSiamese
from commons.metrics import generate_parametrized_qs, global_entanglement_bipartitions
from math import tan
from commons.models.separators import FancySeparator
from commons.pytorch_utils import loc_op_circ, save_acc
import torch

from commons.test_utils.separator import separator_predict


# Gif params
gif_pngs = 21
thresh_bound = (0.0158, 0.02)


# Params
bowl_params = {
    'c': 0.7, #decoherence
    'c_shift': 0.005,
    'a': 1/np.sqrt(2), #amplitude
    'a_shift': 0.005,
    'phi': 0., #phase
    'walls': 'phi', #axis for walls height alignement
    'floor_rad': 8, #radius of the floor in the units of shift params
    'angle': 0.88*np.pi/2, #angle between walls and floor (0 - horizontal alignement, pi/2 - vertical) 
    'angle_shift': 0.0*np.pi/2. # shift of the angle (if equal to 0 const angle will be applied)
}



qubits_num = 3

#----------------------------------------------------------------------
# Separator params
#thresh = 0.0004
out_channels_per_ratio = 24
input_channels = 2
fc_layers = 4
criterion = 'L1'

sep_save_path = './paper_models/3qbits/FancySeparator_l1_all_sep_o48_fc4_bl.pt'
count_save_path = './results/discord/prediction_thresh_param_cuta_sep_param_bl.txt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


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
        if val >= 0.4 and val <= 0.9:
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

step = (thresh_bound[1] - thresh_bound[0]) / gif_pngs
thresholds = np.arange(thresh_bound[0], thresh_bound[1] + step, step)

save_acc(count_save_path, "Threshold", accuracies=["Pred/ZeroDisc", "Prec_ZeroDisc", "Recall_ZeroDisc", "Pred/Separable", "Prec_Separable", "Recall_Separable"])

for gif_num, thresh in enumerate(thresholds):
    print('Current threshold = {}'.format(round(thresh, 4)))

    counter = {
        "zero_discord" : 0,
        "separable" : 0,
        "predicted_zero" : 0,
        "ZDaPZ" : 0,
        "SaPZ" : 0
    }

    fig, ax = plt.subplots()
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

            rho = generate_parametrized_qs(qubits_num, pos['a'], pos['c'], pos['phi'])

            torch_rho = torch.unsqueeze(torch.tensor(rho.data), dim=0)
            torch_data = torch.stack((torch_rho.real, torch_rho.imag), dim= 1)

            torch_data = loc_op_circ(torch_data).double()    
            prediction = separator_predict(model, device, torch_data, thresh, criterion).item()

            neg = global_entanglement_bipartitions(rho, "negativity")
            disc = global_entanglement_bipartitions(rho, "discord")
            
            if prediction >= 0.5:
                color = 'blue'
            else:
                color = 'red'
                counter['predicted_zero'] += 1

            old_angle = bowl_params['angle']
            n_pos_x = next_pos(next_pos(pos, 0, i), 0, i + 1)
            bowl_params['angle'] = old_angle
            n_pos_y = next_pos(next_pos(pos, 1, j), 1, j + 1)
            bowl_params['angle'] = old_angle

            if neg < 0.001:
                counter['separable'] += 1
        
                if check_bounds(floor_axis[0], n_pos_x[floor_axis[0]]):
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_x['a'], n_pos_x['c'], n_pos_x['phi'])
                    next_neg = global_entanglement_bipartitions(next_rho, "negativity")
                    if next_neg > 0.001:
                        color = 'black'

                if check_bounds(floor_axis[1], n_pos_y[floor_axis[1]]):
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_y['a'], n_pos_y['c'], n_pos_y['phi'])
                    next_neg = global_entanglement_bipartitions(next_rho, "negativity")
                    if next_neg > 0.001:
                        color = 'black'
            else:
                if check_bounds(floor_axis[0], n_pos_x[floor_axis[0]]):
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_x['a'], n_pos_x['c'], n_pos_x['phi'])
                    next_neg = global_entanglement_bipartitions(next_rho, "negativity")
                    if next_neg <= 0.001:
                        color = 'black'

                if check_bounds(floor_axis[1], n_pos_y[floor_axis[1]]):
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_y['a'], n_pos_y['c'], n_pos_y['phi'])
                    next_neg = global_entanglement_bipartitions(next_rho, "negativity")
                    if next_neg <= 0.001:
                        color = 'black'
               
            if disc < 0.001:
                counter['zero_discord'] += 1

                if check_bounds(floor_axis[0], n_pos_x[floor_axis[0]]):
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_x['a'], n_pos_x['c'], n_pos_x['phi'])
                    next_disc = global_entanglement_bipartitions(next_rho, "discord")
                    if next_disc > 0.001:
                        color = 'green'

                if check_bounds(floor_axis[1], n_pos_y[floor_axis[1]]):
                    next_rho = generate_parametrized_qs(qubits_num, n_pos_y['a'], n_pos_y['c'], n_pos_y['phi'])
                    next_disc = global_entanglement_bipartitions(next_rho, "discord")
                    if next_disc > 0.001:
                        color = 'green'
 
            if disc < 0.001 and prediction < 0.5:
                counter['ZDaPZ'] += 1
            if neg < 0.001 and prediction < 0.5:
                counter['SaPZ'] += 1


            ax.scatter(pos[floor_axis[0]], pos[floor_axis[1]], c=color, marker = 's', s=5)

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
    lgd = fig.legend(handles=[red_patch, blue_patch, green_patch, black_patch], bbox_to_anchor = (0.1, 0.1))

    x_shift = bowl_params[floor_axis[0] + '_shift'] * bowl_params['floor_rad']
    xs = np.arange(min_x, max_x + x_shift, x_shift)
    x_ticks = [str(round(x, 2)) + " + " + str(round(find_closest_val(x, bowl_params[floor_axis[1]], xy_to_z), 2)) for x in xs]
    plt.xticks(xs, x_ticks, rotation = 'vertical')

    y_shift = bowl_params[floor_axis[1] + '_shift'] * bowl_params['floor_rad']
    ys = np.arange(min_y, max_y + y_shift, y_shift)
    y_ticks = [str(round(y, 2)) + " + " + str(round(find_closest_val(bowl_params[floor_axis[0]], y, xy_to_z), 2)) for y in ys]
    plt.yticks(ys, y_ticks)

    ax.set_xlabel(display_str(floor_axis[0]) + " + " + display_str(z))
    ax.set_ylabel(display_str(floor_axis[1]) + " + " + display_str(z))
    
    plt.savefig("./plots/gif/bowl_cuta_sep_param/discord_bowl_map3q_pred_sep_{}.png".format(gif_num + 79), bbox_extra_artists=[lgd], pad_inches = 0.2, bbox_inches='tight')
    plt.close()

    save_acc(count_save_path, thresh, [counter['ZDaPZ'] / (counter['zero_discord'] + counter['predicted_zero'] - counter['ZDaPZ'] + 1.e-7),
    counter['ZDaPZ']/(counter['predicted_zero'] + 1.e-7), counter['ZDaPZ']/(counter['zero_discord'] + 1.e-7),
    counter['SaPZ'] / (counter['separable'] + counter['predicted_zero'] - counter['ZDaPZ'] + 1.e-7), counter['SaPZ'] / (counter['predicted_zero'] + 1.e-7), 
    counter['SaPZ'] / (counter['separable'] + 1.e-7)])