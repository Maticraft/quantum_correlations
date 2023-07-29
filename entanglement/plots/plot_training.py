import matplotlib.pyplot as plt
import numpy as np


data1 = np.loadtxt('c:/studia/PhD/entanglement_detection/results/vectorClassifier_3q_nopptes_training_val_balb.txt', delimiter= "  ")

plt.grid(True)
plt.xlabel("Epoch")
plt.ylabel('BCE loss')
plt.plot(data1[:, 0], data1[:, 1], 'o-.', label='training set')
plt.plot(data1[:, 0], data1[:, 3], 's-.', label='validation set')
plt.legend()
plt.xticks([int(x) for x in data1[:, 0]])
plt.savefig('./plots/final_plots/trainingCNN_3q_val_bal_loss.png')
plt.close()

plt.grid(True)
plt.xlabel("Epoch")
plt.ylabel('Accuracy [%]')
plt.plot(data1[:, 0], data1[:, 2], 'o-.', label='training set')
plt.plot(data1[:, 0], data1[:, 4], 's-.', label='validation set')
plt.legend()
plt.xticks([int(x) for x in data1[:, 0]])
plt.savefig('./plots/final_plots/trainingCNN_3q_val_bal_acc.png')
plt.close()