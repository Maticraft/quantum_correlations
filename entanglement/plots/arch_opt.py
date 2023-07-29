import matplotlib.pyplot as plt
import numpy as np


data1 = np.loadtxt('./results/article_test_siam_opt_c2.txt', delimiter='  ', skiprows=1)
data2 = np.loadtxt('./results/article_test_siam_opt_c3.txt', delimiter='  ', skiprows=1)

data3 = np.loadtxt('./results/article_test_cnn_opt_c2.txt', delimiter='  ', skiprows=1)
data4 = np.loadtxt('./results/article_test_cnn_opt_c3.txt', delimiter='  ', skiprows=1)

plt.grid(True)
fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].set_title('CNN model')
axs[0].set_xlabel("Kernel size")
axs[0].set_ylabel("Accuracy [%]")
axs[0].plot(data1[:, 0], data3[:, 2], 's--', label='2 conv layers')
axs[0].plot(data2[:, 0], data4[:, 2], 'x--', label='3 conv layers')
axs[1].set_title('Siamese network model')
axs[1].set_xlabel("Kernel size")
axs[1].plot(data1[:, 0], data1[:, 2], 's--', label='2 conv layers')
axs[1].plot(data2[:, 0], data2[:, 2], 'x--', label='3 conv layers')

lbls = axs[0].get_legend_handles_labels()
lgd = axs[1].legend(*lbls, loc = 'lower right')
plt.tight_layout(pad = 1)
plt.savefig('./plots/final_plots/arch_opt.png', bbox_extra_artists=[lgd], bbox_inches='tight')
plt.close()
