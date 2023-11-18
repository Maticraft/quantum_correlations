from math import tan

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator

from commons.data.generation_functions import generate_parametrized_qs
from commons.metrics import global_entanglement_bipartitions


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
    else:
        if val >= 0. and val <= 1.:
            return True
        else:
            return False


# Params
bowl_params = {
    'c': 0.7, #decoherence
    'c_shift': 0.02,
    'a': 1/np.sqrt(2), #amplitude
    'a_shift': 0.02,
    'phi': 0., #phase
    'phi_shift': 0.1,
    'walls': 'phi', #axis for walls height alignement
    'floor_rad': 2, #radius of the floor in the units of shift params
    'angle': 0.9*np.pi/2 #angle between walls and floor (0 - horizontal alignement, pi/2 - vertical) 
}

qubits_num = 3

#----------------------------------------------------------------------
# Execution

def floor_to_z_np(floor_param, np_val):
    return np.array([floor_to_z(floor_param, val) for val in np_val])


def floor_to_z(floor_param, val):
    high_boundary = bowl_params[floor_param] + bowl_params['floor_rad'] * bowl_params[floor_param + '_shift']
    low_boundary = bowl_params[floor_param] - bowl_params['floor_rad'] * bowl_params[floor_param + '_shift']

    if val >= low_boundary and val <= high_boundary:
        return 0.
    elif val > high_boundary:
        return tan(bowl_params['angle']) * (val - high_boundary)
    else:
        return tan(bowl_params['angle']) * (low_boundary - val)



fig, ax = plt.subplots()

z = bowl_params['walls']
floor_axis = [k for k in ['a','c','phi'] if k != z]

pos = {
        floor_axis[0]: bowl_params[floor_axis[0]],
        floor_axis[1]: bowl_params[floor_axis[1]],
        z: bowl_params[z]  
    }

pos_zi = pos[z]
min_x, max_x = 1., 0
min_y, max_y = 1., 0.

i = 0
while check_bounds(floor_axis[0], pos[floor_axis[0]]):
    print("Current pos:\n {}".format(pos))
    if pos[floor_axis[0]] < min_x:
        min_x = pos[floor_axis[0]]
    if pos[floor_axis[0]] > max_x:
        max_x = pos[floor_axis[0]]

    pos[floor_axis[1]] = bowl_params[floor_axis[1]]
    pos[z] = pos_zi

    j = 0
    while check_bounds(floor_axis[1], pos[floor_axis[1]]):
        if pos[floor_axis[1]] > max_y:
            max_y = pos[floor_axis[1]]
        if pos[floor_axis[1]] < min_y:
            min_y = pos[floor_axis[1]]

        rho = generate_parametrized_qs(qubits_num, pos['a'], pos['c'], pos['phi'])

        # Optional randomizing density matrix with single qubit operations 
        #rho = local_randomize_matrix(np.arange(qubits_num), rho, num_gates = 3)
        neg = global_entanglement_bipartitions(rho, "negativity")
        disc = global_entanglement_bipartitions(rho, "discord")

        if neg < 0.001:
            color = 'red'
        else:
            color = 'blue'
        
        if disc < 0.001:
            color = "green"
            marker = 'o'
        else:
            marker = 'v'

        pos_zj = pos[z] - pos_zi + bowl_params[z]
        ax.scatter(pos[floor_axis[0]], pos[floor_axis[1]], c=color, marker=marker)

        if j % 2 == 0:    
            if j / 2 > bowl_params['floor_rad']:
                pos[z] = shift_param(z, bowl_params[z + '_shift'], pos[z])
                pos[floor_axis[1]] = shift_param(floor_axis[1], (1/tan(bowl_params['angle'])) * bowl_params[z + '_shift'], pos[floor_axis[1]])
            else:
                pos[floor_axis[1]] = shift_param(floor_axis[1], bowl_params[floor_axis[1] + '_shift'], pos[floor_axis[1]])

        pos[floor_axis[1]] = 2 * bowl_params[floor_axis[1]] - pos[floor_axis[1]]
        j += 1

    if i % 2 == 0:
        if i / 2 > bowl_params['floor_rad']:
            pos_zi = shift_param(z, bowl_params[z + '_shift'], pos_zi)
            pos[floor_axis[0]] = shift_param(floor_axis[0], (1/tan(bowl_params['angle'])) * bowl_params[z + '_shift'], pos[floor_axis[0]])
        else:
            pos[floor_axis[0]] = shift_param(floor_axis[0], bowl_params[floor_axis[0] + '_shift'], pos[floor_axis[0]])

    pos[floor_axis[0]] = 2 * bowl_params[floor_axis[0]] - pos[floor_axis[0]]
    i += 1
      

red_patch = mpatches.Patch(color='red', label='separable')
blue_patch = mpatches.Patch(color='blue', label='entangled')
green_patch = mpatches.Patch(color='green', label='zero discord')
lgd = fig.legend(handles=[red_patch, blue_patch, green_patch], bbox_to_anchor = (0.96, 0.95))

xs = np.arange(min_x, max_x, bowl_params[floor_axis[0] + '_shift'] * 2)
x_ticks = [str(round(x, 2)) + " + " + str(round(floor_to_z(floor_axis[0], x), 2)) for x in xs]
plt.xticks(xs, x_ticks, rotation = 'vertical')

ys = np.arange(min_y, max_y, bowl_params[floor_axis[1] + '_shift'] * 2)
y_ticks = [str(round(y, 2)) + " + " + str(round(floor_to_z(floor_axis[0], y), 2)) for y in ys]
plt.yticks(ys, y_ticks)

ax.set_xlabel(str(floor_axis[0]) + " + " + str(z))
ax.set_ylabel(str(floor_axis[1]) + " + " + str(z))

plt.tight_layout(pad = 2)

plt.savefig("./plots/discord_bowl_map{}q.png".format(qubits_num))