import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('./results/discord/prediction_thresh_mixed_sep_param_bl_bal_acc_log.txt', delimiter='  ', skiprows=1)


fig, axs = plt.subplots(2, 2)
axs[1, 0].grid(True)
axs[1, 0].set_xscale('log')
axs[1, 0].set_ylim(0, 1.)
axs[1, 0].set_title("Jaccard Similarity")
axs[1, 0].set_xlabel("Threshold")
axs[1, 0].plot(data[:, 0], data[:, 1], 'o--', markersize = 3, linewidth = 1, label='Zero-discord states')
axs[1, 0].plot(data[:, 0], data[:, 4], 'x--', markersize = 4, linewidth =1,  label='Separable states')

axs[1, 1].grid(True)
axs[1, 1].set_xscale('log')
axs[1, 1].set_ylim(0, 1.)
axs[1, 1].set_title("Balanced Accuracy")
axs[1, 1].set_xlabel("Threshold")
axs[1, 1].plot(data[:, 0], data[:, 7], 'o--', markersize = 3, linewidth = 1, label='Zero-discord states')
axs[1, 1].plot(data[:, 0], data[:, 8], 'x--', markersize = 4, linewidth =1,  label='Separable states')

axs[0, 0].set_xscale('log')
axs[0, 0].grid(True)
axs[0, 0].set_title("Precision")
axs[0, 0].set_xlabel("Threshold")
axs[0, 0].plot(data[:, 0], data[:, 2], 'o--', markersize = 3, linewidth = 1, label='Zero-discord states')
axs[0, 0].plot(data[:, 0], data[:, 5], 'x--', markersize = 4, linewidth =1,  label='Separable states')

axs[0, 1].set_xscale('log')
axs[0, 1].grid(True)
axs[0, 1].set_title("Recall")
axs[0, 1].set_xlabel("Threshold")
axs[0, 1].plot(data[:, 0], data[:, 3], 'o--', markersize = 3, linewidth = 1, label='Zero-discord states')
axs[0, 1].plot(data[:, 0], data[:, 6], 'x--', markersize = 4, linewidth = 1, label='Separable states')


lbls = axs[0, 0].get_legend_handles_labels()
lgd = fig.legend(*lbls, bbox_to_anchor = (0.65, 0.01))
plt.tight_layout(pad = 0.7)
plt.savefig('./plots/similarity_neg_disc_log2.eps', bbox_extra_artists=[lgd], bbox_inches='tight')
plt.close()






