import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('./results/discord/confusion_acc_std_fancy_train_mixed_sep_test_mixed_avg17.txt', delimiter='  ', skiprows=1)

x1 = np.logspace(-4, -3, 50)
x2 = np.logspace(-3, -1, 50)
y1 = 100*(-555.5555555555555*x1 + 0.6055555555555555) + 45
y2 = 100*(5.05050505050505*x2 + 0.04494949494949495) + 45
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))

plt.grid(True)
plt.xlabel("Threshold")
plt.ylabel('Accuracy [%]')
plt.xscale('log')
#plt.ylim(50, 105)
plt.xlim(1.e-4, 0.04)
plt.plot(data[:, 0], data[:, 2], 's--', label='Best accuracy model')
plt.plot(data[:, 0], data[:, 4], 'x--', label='Best loss model')
plt.plot(x, y, 'r--', label='Random guess')
plt.legend()
plt.savefig('./plots/confusion_acc_std_fancy_train_mixed_sep_test_mixed_avg17.png')
plt.close()
