import matplotlib.pyplot as plt
import numpy as np

data1 = np.loadtxt('./results/article_test_mixed_def_ppt_circ_n_4q_acc_vec_valb_nopptes.txt', delimiter='  ', skiprows=1)
data2 = np.loadtxt('./results/article_test_mixed_def_ppt_haar_n_4q_acc_vec_valb_nopptes.txt', delimiter='  ', skiprows=1)
data3 = np.loadtxt('./results/article_fullent_dens_4q_circ.txt', delimiter=' ', skiprows=1)
data4 = np.loadtxt('./results/article_fullent_dens_4q_haar.txt', delimiter=' ', skiprows=1)


data5 = np.loadtxt('./results/article_test_mixed_def_ppt_circ_n_4q_acc_vec_valb.txt', delimiter='  ', skiprows=1)
data6 = np.loadtxt('./results/article_test_mixed_def_ppt_haar_n_4q_acc_vec_valb.txt', delimiter='  ', skiprows=1)

#1st column for NPT acc
#3rd column for total acc

fig, axs = plt.subplots(1, 2)
axs[0].grid(True)
axs[0].set_title("Circuit-generated states")
axs[0].set_xlabel("Number of pure states")
axs[0].plot(data5[:, 0], data5[:, 1], 's--', label='Falsely labeled model accuracy [frac]')
axs[0].plot(data1[:, 0], data1[:, 1], 's--', label='No-pptes model accuracy [frac]')
axs[0].plot(data3[:, 0], data3[:, 1], 'x--', label='Fraction of all states, which are NPT')

axs[1].grid(True)
axs[1].set_title("Haar-generated states")
axs[1].set_xlabel("Number of pure states")
axs[1].plot(data6[:, 0], data6[:, 1], 's--', label='Falsely labeled model accuracy [frac]')
axs[1].plot(data2[:, 0], data2[:, 1], 's--', label='No-pptes model accuracy [frac]')
axs[1].plot(data4[:, 0], data4[:, 1], 'x--', label='Fraction of all states, which are NPT')


lbls = axs[0].get_legend_handles_labels()
lgd = fig.legend(*lbls, bbox_to_anchor = (0.67, 0.05))
plt.tight_layout(pad = 3)
plt.savefig('./plots/final_plots/mixed_n_def_4q_valb.png', bbox_extra_artists=[lgd], bbox_inches='tight')
plt.close()






