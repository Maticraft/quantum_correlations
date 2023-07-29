import matplotlib.pyplot as plt
import numpy as np

qubits_num = 4
xrange = (1, 120)
interval = 10

data1 = np.loadtxt(f'./results/article_test_mixed_def_sep_n_{qubits_num}q_acc_vec_valb.txt', delimiter='  ', skiprows=1)
data2 = np.loadtxt(f'./results/article_test_mixed_def_sep_n_{qubits_num}q_acc_vec_valb_nopptes.txt', delimiter='  ', skiprows=1)


plt.grid(True)
plt.xlabel("Number of pure states")
plt.ylabel('Accuracy [frac]')
plt.xlim(*xrange)
plt.plot(data1[:, 0], data1[:, 4]/100, 's--', markersize = 3, linewidth = 1, label='verified model')
plt.plot(data2[:, 0], data2[:, 4]/100, 'o--', markersize = 3, linewidth = 1, label='weakly trained model')
plt.xticks(np.insert(np.arange(interval, xrange[1] + interval, interval), 0, 1))
plt.legend()
plt.tight_layout(pad = 3)
plt.savefig(f'./plots/final_plots/acc_mixed_sep_n_{qubits_num}q.png')
plt.close()
