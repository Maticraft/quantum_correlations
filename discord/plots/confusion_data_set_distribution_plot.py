import matplotlib.pyplot as plt
import numpy as np


with open('./results/discord/separator_loss_mixed_sep_test.txt', 'r') as f:
    data = f.readlines()
    data = [float(x.strip().split(', ')[1]) for x in data]


logbins = np.geomspace(min(data), max(data), 100)

counts = np.histogram(data, bins=logbins)[0] / len(data)
plt.bar(logbins[:-1], counts, width=np.diff(logbins))

plt.grid(True)
plt.title('Data distribution')
plt.xlabel("Threshold")
plt.xscale('log')
plt.ylabel('Density')
plt.savefig('./plots/confusion_data_mixed_sep_test_distrib.png')
plt.close()
