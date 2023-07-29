import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from commons.metrics import generate_parametrized_qs, global_entanglement_bipartitions, local_randomize_matrix

# Params
a = 1/np.sqrt(2) #amplitude
cs = np.arange(0, 1 + 0.05, 0.05) # decoherence
fis = np.append(np.arange(0., 2*np.pi, 0.1), 2*np.pi) # phase
qubits_num = 2

fig = plt.figure()

for c in cs:
    print("Current c = {}".format(c))
    for fi in fis:
        rho = generate_parametrized_qs(qubits_num, a, c, fi)
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

        plt.scatter(fi, c, c=color, marker=marker)

red_patch = mpatches.Patch(color='red', label='separable')
blue_patch = mpatches.Patch(color='blue', label='entangled')
green_patch = mpatches.Patch(color='green', label='zero discord')
lgd = fig.legend(handles=[red_patch, blue_patch, green_patch], bbox_to_anchor = (0.96, 0.95))
  
plt.xlabel(r"$\phi$")
plt.ylabel("c")
plt.tight_layout(pad = 2)

plt.savefig("./discord_map2q_shift.png")