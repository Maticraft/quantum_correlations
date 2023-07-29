import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

data = np.loadtxt('articles_test_pptes.txt', delimiter='  ')
data = np.round(data, 2)

fig, axs = plt.subplots(1, 3, sharey=True)
fig.set_figheight(4)
fig.set_figwidth(12)

axs[0].set_title("Horodecki PPTES", y = -0.1)
axs[0].set_ylabel("Accuracy [%]")
axs[0].set_xlim([1, 5])
axs[0].bar([2, 4] , data[:, 0],  color= ['orange', 'blue'], width = 0.8)
axs[0].text(2, data[0, 0] + 1, data[0, 0], ha='center')
axs[0].text(4, data[1, 0] + 1, data[1, 0], ha='center')
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].get_xaxis().set_visible(False)

axs[1].set_title("Bennet PPTES", y = -0.1)
axs[1].set_xlim([1, 5])
axs[1].bar([2, 4] , data[:, 1],  color= ['orange', 'blue'])
axs[1].text(2, data[0, 1] + 1, data[0, 1], ha='center')
axs[1].text(4, data[1, 1] + 1, data[1, 1], ha='center')
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_visible(False)
axs[1].get_yaxis().set_visible(False)
axs[1].get_xaxis().set_visible(False)

axs[2].set_title("Acin PPTES", y = -0.1)
axs[2].set_xlim([1, 5])
axs[2].bar([2, 4] , data[:, 2],  color= ['orange', 'blue'])
axs[2].text(2, data[0, 2] + 1, data[0, 2], ha='center')
axs[2].text(4, data[1, 2] + 1, data[1, 2], ha='center')
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].spines['left'].set_visible(False)
axs[2].get_yaxis().set_visible(False)
axs[2].get_xaxis().set_visible(False)

orange_patch = mpatches.Patch(color='orange', label='Bipartition accuracy')
blue_patch = mpatches.Patch(color='blue', label='Global accuracy')
lgd = fig.legend(handles=[orange_patch, blue_patch], bbox_to_anchor = (0.6, 0.05))
  
plt.tight_layout(pad = 3)
plt.savefig('./plots/acc_pptes.png', bbox_extra_artists=[lgd], bbox_inches='tight')
plt.close()
