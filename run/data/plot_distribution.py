import matplotlib.pyplot as plt
import numpy as np


distributions_path = './datasets/3qbits/train_bisep_no_pptes/bisep_distribution.txt'
color_entangled = 'xkcd:blue'
color_separable = 'xkcd:red'
color_biseparable = 'xkcd:green'

# load data to numpy ignoring the first column
with open(distributions_path, 'r') as f:
    data = f.readlines()

data = [x.strip().split(', ')[1:] for x in data]
data = np.array(data, dtype=np.float64)

bins = np.geomspace(0.0001, 0.04, 30)
data_bisep = data[:, 0][data[:, 1] == 2]
p2, x2 = np.histogram(data_bisep, bins=bins)
p2 = p2 / p2.sum()

bins = np.geomspace(0.0001, 0.04, 30)
data_sep = data[:, 0][data[:, 1] == 0]
p0, x0 = np.histogram(data_sep, bins=bins)
p0 = p0 / p0.sum()

bins = np.geomspace(0.0001, 0.04, 30)
data_ent = data[:, 0][data[:, 1] == 1]
p1, x1 = np.histogram(data_ent, bins=bins)
p1 = p1 / p1.sum()

plt.rcParams.update({'font.size': 18})

plt.figure()
plt.plot(x0[:-1], p0, label='Separable states', color=color_separable)
plt.fill_between(x0[:-1], p0, alpha=0.3, color=color_separable)
plt.plot(x2[:-1], p2, label='Biseparable states', color=color_biseparable)
plt.fill_between(x2[:-1], p2, alpha=0.3, color=color_biseparable)
plt.plot(x1[:-1], p1, label='Entangled states', color=color_entangled)
plt.fill_between(x1[:-1], p1, alpha=0.3, color=color_entangled)
# plt.title('Separability metrics (separator) distribution')
plt.xlabel(r'Separability measure $M$')
plt.xscale('log')
plt.ylabel('Distribution')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/train_separator_loss_bisep_distribution_log.png')
plt.close()


# bins = np.geomspace(0.0001, 0.1, 30)
# data_sep = data[:, 0][data[:, 1] == 0]
# p, x = np.histogram(data_sep, bins=bins)
# p = p / p.sum()

# plt.figure()
# plt.plot(x[:-1], p)
# plt.title('Separability metrics (separator) distribution')
# plt.xlabel('Loss')
# plt.xscale('log')
# plt.ylabel('Distribution')
# plt.savefig('./plots/train_separator_loss_separable_distribution.png')
# plt.close()


# bins = np.geomspace(0.0001, 0.1, 30)
# data_ent = data[:, 0][data[:, 1] == 1]
# p, x = np.histogram(data_ent, bins=bins)
# p = p / p.sum()

# plt.figure()
# plt.plot(x[:-1], p)
# plt.title('Separability metrics (separator) distribution')
# plt.xlabel('Loss')
# plt.xscale('log')
# plt.ylabel('Distribution')
# plt.savefig('./plots/train_separator_loss_entangled_distribution.png')
# plt.close()


bins = np.linspace(0, 2.1, 4) 
p, x = np.histogram(data[:,1], bins=bins)
# p = p / p.sum()

plt.figure()
plt.bar([0, 1, 2], height=p, width=0.5)
# plt.title('Target distribution')
plt.xlabel('Target')
plt.xticks([0, 1, 2], ['Separable', 'Entangled', 'Biseparable']) # what about biseparable?
plt.ylabel('Distribution')
plt.savefig('./plots/target_bisep_distribution.png')
plt.close()


bins = np.geomspace(0.001, 1, 30)
data_ent = data[:, 2][data[:, 1] == 1]
p1, x1 = np.histogram(data_ent, bins=bins)
p1 = p1 / p1.sum()

bins = np.geomspace(0.001, 1, 30)
data_bisep = data[:, 2][data[:, 1] == 2]
p2, x2 = np.histogram(data_bisep, bins=bins)
p2 = p2 / p2.sum()

plt.figure()
plt.plot(x2[:-1], p2, label='Biseparable states', color=color_biseparable)
plt.fill_between(x2[:-1], p2, alpha=0.3, color=color_biseparable)
plt.plot(x1[:-1], p1, label='Entangled states', color=color_entangled)
plt.fill_between(x1[:-1], p1, alpha=0.3, color=color_entangled)
# plt.title('Negativity distribution')
plt.xlabel('Negativity')
plt.ylabel('Distribution')
plt.legend()
plt.xscale('log')
plt.tight_layout()
plt.savefig('./plots/negativity_bisep_distribution_log.png')
plt.close()


# bins = np.linspace(0.001, 1, 30)
# data_neg = data[:, 2][data[:, 1] == 2]
# p, x = np.histogram(data_neg, bins=bins)
# p = p / p.sum()

# plt.figure()
# plt.plot(x[:-1], p)
# plt.title('Negativity distribution')
# plt.xlabel('Negativity')
# plt.ylabel('Distribution')
# plt.savefig('./plots/negativity_bisep_distribution.png')
# plt.close()
