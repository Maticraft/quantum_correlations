import matplotlib.pyplot as plt
import numpy as np

qubits_num = 3
xrange = (1, 100)
interval = 10

data1 = np.loadtxt(f'./results/article_test_mixed_def_ppt_circ_n_{qubits_num}q_acc_vec_valb_nopptes.txt', delimiter='  ', skiprows=1)
data2 = np.loadtxt(f'./results/article_test_mixed_def_ppt_haar_n_{qubits_num}q_acc_vec_valb_nopptes.txt', delimiter='  ', skiprows=1)
data3 = np.loadtxt(f'./results/article_fullent_dens_{qubits_num}q_circ.txt', delimiter=' ', skiprows=1)
data4 = np.loadtxt(f'./results/article_fullent_dens_{qubits_num}q_haar.txt', delimiter=' ', skiprows=1)


data5 = np.loadtxt(f'./results/article_test_mixed_def_ppt_circ_n_{qubits_num}q_acc_vec_valb.txt', delimiter='  ', skiprows=1)
data6 = np.loadtxt(f'./results/article_test_mixed_def_ppt_haar_n_{qubits_num}q_acc_vec_valb.txt', delimiter='  ', skiprows=1)

#1st column for NPT acc
#3rd column for total acc

fig, ax = plt.subplots(1, 1)
ax.grid(True)
ax.set_xlabel("Number of pure states")
ax.set_xlim(*xrange)
ax.plot(data5[:, 0], data5[:, 3]/100, 's--', markersize = 3, linewidth = 1, label='NegConv for weakly trained model [frac]')
ax.plot(data1[:, 0], data1[:, 3]/100, 'o--', markersize = 3, linewidth = 1, label='NegConv for verified model [frac]')
ax.plot(data3[:, 0], data3[:, 1], 'x--', markersize = 4, linewidth =1,  label='Fraction of all states, which are NPT')
ax.set_xticks(np.insert(np.arange(interval, xrange[1] + interval, interval), 0, 1))

lbls = ax.get_legend_handles_labels()
lgd = fig.legend(bbox_to_anchor = (0.93, 0.05))
plt.tight_layout(pad = 3)
plt.savefig(f'./plots/final_plots/mixed_n_def_{qubits_num}q_valb_all_circ.png', bbox_extra_artists=[lgd], bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(1, 1)
ax.grid(True)
ax.set_xlabel("Number of pure states")
ax.set_xlim(*xrange)
ax.plot(data6[:, 0], data6[:, 3]/100, 's--', markersize = 3, linewidth = 1, label='NegConv for weakly trained model [frac]')
ax.plot(data2[:, 0], data2[:, 3]/100, 'o--', markersize = 3, linewidth = 1, label='NegConv for verified model [frac]')
ax.plot(data4[:, 0], data4[:, 1], 'x--', markersize = 4, linewidth = 1, label='Fraction of all states, which are NPT')
ax.set_xticks(np.insert(np.arange(interval, xrange[1] + interval, interval), 0, 1))

lbls = ax.get_legend_handles_labels()
lgd = fig.legend(bbox_to_anchor = (0.93, 0.05))
plt.tight_layout(pad = 3)
plt.savefig(f'./plots/final_plots/mixed_n_def_{qubits_num}q_valb_all_haar.png', bbox_extra_artists=[lgd], bbox_inches='tight')
plt.close()






