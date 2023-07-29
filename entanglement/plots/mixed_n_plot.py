import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt('article_test_mixed_def_n_3q_acc_vec.txt', delimiter='  ', skiprows=1)

plt.grid(True)
plt.xlabel("Number of pure states")
plt.ylabel('Accuracy [%]')
plt.plot(data[:, 0], data[:, 1], 's--', label='Entangled states')
plt.plot(data[:, 0], data[:, 2], 'x--', label='Separable states')
plt.legend()
plt.xticks(np.arange(1, 21, 2))
plt.savefig('./plots/acc_mixed_n.png')
plt.close()
