import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

#siam for siamese network
#cnn 
data = np.loadtxt('./results/article_test_cnn_scaling_acc_vec_valb_ext.txt', delimiter='  ', skiprows=1)

data = np.round(data, 2)

plt.rc('font', size=15) #controls default text size
plt.rc('axes', titlesize=17) #fontsize of the title
plt.rc('axes', labelsize=15) #fontsize of the x and y labels
plt.rc('xtick', labelsize=15) #fontsize of the x tick labels
plt.rc('ytick', labelsize=15) #fontsize of the y tick labels
plt.rc('legend', fontsize=15) #fontsize of the legend

fig, axs = plt.subplots(1, 3, sharey=True)
fig.set_figheight(4)
fig.set_figwidth(12)

axs[0].set_title("Pure states", y = -0.15)
axs[0].set_ylabel("Accuracy [%]")
axs[0].set_xlim([1, 4])
axs[0].set_ylim([80, 100])
axs[0].bar([1.5, 2.5, 3.5] , data[:, 1],  color= ['orange', 'blue', 'green'], width = 0.8)
axs[0].text(1.5, data[0, 1] + 0.5, data[0, 1], ha='center')
axs[0].text(2.5, data[1, 1] + 0.5, data[1, 1], ha='center')
axs[0].text(3.5, data[2, 1] + 0.5, data[2, 1], ha='center')
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].get_xaxis().set_visible(False)

axs[1].set_title("Mixed states", y = -0.15)
axs[1].set_xlim([1, 4])
axs[1].set_ylim([80, 100])
axs[1].bar([1.5, 2.5, 3.5] , data[:, 2],  color= ['orange', 'blue', 'green'], width = 0.8)
axs[1].text(1.5, data[0, 2] + 0.5, data[0, 2], ha='center')
axs[1].text(2.5, data[1, 2] + 0.5, data[1, 2], ha='center')
axs[1].text(3.5, data[2, 2] + 0.5, data[2, 2], ha='center')
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_visible(False)
axs[1].get_yaxis().set_visible(False)
axs[1].get_xaxis().set_visible(False)

axs[2].set_title("Extended Horodecki PPTES", y = -0.15)
axs[2].set_xlim([1, 4])
axs[2].set_ylim([80, 100])
axs[2].bar([1.5, 2.5, 3.5] , data[:, 4],  color= ['orange', 'blue', 'green'], width = 0.8)
axs[2].text(1.5, data[0, 4] + 0.5, data[0, 4], ha='center')
axs[2].text(2.5, data[1, 4] + 0.5, data[1, 4], ha='center')
axs[2].text(3.5, data[2, 4] + 0.5, data[2, 4], ha='center')
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].spines['left'].set_visible(False)
axs[2].get_yaxis().set_visible(False)
axs[2].get_xaxis().set_visible(False)

orange_patch = mpatches.Patch(color='orange', label='3 qubits')
blue_patch = mpatches.Patch(color='blue', label='4 qubits')
green_patch = mpatches.Patch(color='green', label='5 qubits')
lgd = fig.legend(handles=[orange_patch, blue_patch, green_patch], bbox_to_anchor = (1.12, 0.65))
  
plt.tight_layout(pad = 2)
plt.savefig('./plots/final_plots/cnn_scaling_nopptes.png', bbox_extra_artists=[lgd], pad_inches = 0.3, bbox_inches='tight')
plt.close()
