import sys
sys.path.append('./')
import os
import matplotlib.pyplot as plt
from commons.pytorch_utils import load_acc

data_path = './datasets/3qbits/train_separable'
save_file_name = 'correlations.txt'
save_path = os.path.join(data_path, save_file_name)

data = load_acc(save_path, skiprows=1)
max_loss = data[:, 0]
loss_diff = data[:, 1]
l2 = data[:, 2]
bures = data[:, 3]

fig = plt.figure()
plt.scatter(max_loss, loss_diff)
plt.xlabel('Max loss')
plt.ylabel('Loss diff')
plt.savefig(os.path.join(data_path, 'max_loss_loss_diff.png'))

fig = plt.figure()
plt.scatter(max_loss, l2)
plt.xlabel('Max loss')
plt.ylabel('L2')
plt.savefig(os.path.join(data_path, 'max_loss_l2.png'))

fig = plt.figure()
plt.scatter(max_loss, bures)
plt.xlabel('Max loss')
plt.ylabel('Bures')
plt.savefig(os.path.join(data_path, 'max_loss_bures.png'))

fig = plt.figure()
plt.scatter(loss_diff, l2)
plt.xlabel('Loss diff')
plt.ylabel('L2')
plt.savefig(os.path.join(data_path, 'loss_diff_l2.png'))

fig = plt.figure()
plt.scatter(loss_diff, bures)
plt.xlabel('Loss diffs')
plt.ylabel('Bures')
plt.savefig(os.path.join(data_path, 'loss_diff_bures.png'))

fig = plt.figure()
plt.scatter(l2, bures)
plt.xlabel('L2')
plt.ylabel('Bures')
plt.savefig(os.path.join(data_path, 'l2_bures.png'))