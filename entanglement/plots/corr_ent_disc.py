import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('./results/discord_sep_acc.txt', delimiter='  ', skiprows=1)


fig, axs = plt.subplots(1, 2)
axs[0].set_xlim([0, 0.04])
axs[0].grid(True)
axs[0].set_title("Accuracy")
axs[0].set_xlabel("Threshold")
axs[0].plot(data[:, 0], data[:, 1], 's--', markersize = 3, linewidth = 1, label='Pure states')
axs[0].plot(data[:, 0], data[:, 3], 'o--', markersize = 3, linewidth = 1, label='Mixed reduced states negativity')
axs[0].plot(data[:, 0], data[:, 5], 'x--', markersize = 4, linewidth =1,  label='Mixed reduced states discord')

axs[1].grid(True)
axs[1].set_xlim([0, 0.04])
axs[1].set_title("Balanced accuracy")
axs[1].set_xlabel("Threshold")
axs[1].plot(data[:, 0], data[:, 2], 's--', markersize = 3, linewidth = 1, label='Pure states')
axs[1].plot(data[:, 0], data[:, 4], 'o--', markersize = 3, linewidth = 1, label='Mixed reduced states negativity')
axs[1].plot(data[:, 0], data[:, 6], 'x--', markersize = 4, linewidth = 1, label='Mixed reduced states discord')


lbls = axs[0].get_legend_handles_labels()
lgd = fig.legend(*lbls, bbox_to_anchor = (0.93, 0.05))
plt.tight_layout(pad = 3)
plt.savefig('./plots/corr_mr_neg_disc.png', bbox_extra_artists=[lgd], bbox_inches='tight')
plt.close()






