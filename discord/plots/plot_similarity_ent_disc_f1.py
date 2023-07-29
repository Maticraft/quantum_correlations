import matplotlib.pyplot as plt
import numpy as np
from commons.pytorch_utils import load_acc

# data1 = np.loadtxt('./results/discord/trace_prediction_thresh_mixed_sep_param_bl_bal_acc_log.txt', delimiter='  ', skiprows=1)
# data2 = np.loadtxt('./results/discord/prediction_thresh_mixed_sep_param_bl_bal_acc_log.txt', delimiter='  ', skiprows=1)

data1 = load_acc('./results/discord/bures_trace_prediction_thresh_mixed_bal_bal_acc_log.txt', skiprows=1)
data2 = load_acc('./results/discord/bures_sep_all_sep_fc4_prediction_thresh_mixed_bal_bal_acc_log.txt', skiprows=1)

# Correct precision if it is equal to 0
data1[:, 2] = np.where(data1[:, 2] == 0, 1., data1[:, 2])
data1[:, 5] = np.where(data1[:, 5] == 0, 1., data1[:, 5])
data2[:, 2] = np.where(data2[:, 2] == 0, 1., data2[:, 2])
data2[:, 5] = np.where(data2[:, 5] == 0, 1., data2[:, 5])

f1_score_zd1 = 2 * data1[:, 2] * data1[:, 3] / (data1[:, 2] + data1[:, 3])
f1_score_sep1 = 2 * data1[:, 5] * data1[:, 6] / (data1[:, 5] + data1[:, 6])

# Zero-discord states
data1_zd = np.array(sorted(data1[:, 2:4], key=lambda x: x[1]))
auc_zd1 = 0
for i in range(1, len(data1_zd)):
    auc_zd1 += (data1_zd[i, 1] - data1_zd[i-1, 1]) * (data1_zd[i, 0] + data1_zd[i-1, 0]) / 2

# Separable states
data1_sep = np.array(sorted(data1[:, 5:7], key=lambda x: x[1]))
auc_sep1 = 0
for i in range(1, len(data1_sep)):
    auc_sep1 += (data1_sep[i, 1] - data1_sep[i-1, 1]) * (data1_sep[i, 0] + data1_sep[i-1, 0]) / 2


f1_score_zd2 = 2 * data2[:, 2] * data2[:, 3] / (data2[:, 2] + data2[:, 3])
f1_score_sep2 = 2 * data2[:, 5] * data2[:, 6] / (data2[:, 5] + data2[:, 6])

# Zero-discord states
data2_zd = np.array(sorted(data2[:, 2:4], key=lambda x: x[1]))
auc_zd2 = 0
for i in range(1, len(data2_zd)):
    auc_zd2 += (data2_zd[i, 1] - data2_zd[i-1, 1]) * (data2_zd[i, 0] + data2_zd[i-1, 0]) / 2

# Separable states
data2_sep = np.array(sorted(data2[:, 5:7], key=lambda x: x[1]))
auc_sep2 = 0
for i in range(1, len(data2_sep)):
    auc_sep2 += (data2_sep[i, 1] - data2_sep[i-1, 1]) * (data2_sep[i, 0] + data2_sep[i-1, 0]) / 2

# Set global font size
plt.rcParams.update({'font.size': 14})


fig, axs = plt.subplots(3, 2, figsize=(10, 8))
axs[0, 0].set_xscale('log')
axs[0, 0].grid(True)
axs[0, 0].set_title("Precision")
axs[0, 0].set_xlabel("Threshold")
axs[0, 0].plot(data1[:, 0], data1[:, 2], 'o--', markersize = 3, linewidth = 1, label='Trace reconstruction of zero-discord states')
axs[0, 0].plot(data1[:, 0], data1[:, 5], 'x--', markersize = 4, linewidth =1,  label='Trace reconstruction of separable states')
axs[0, 0].plot(data2[:, 0], data2[:, 2], 'o--', markersize = 3, linewidth = 1, label='Separator reconstruction of zero-discord states')
axs[0, 0].plot(data2[:, 0], data2[:, 5], 'x--', markersize = 4, linewidth =1,  label='Separator reconstruction of separable states')
# Mark subplot with letter over the plot
axs[0, 0].text(0.05, 0.8, '(a)', transform=axs[0, 0].transAxes, fontsize=14, fontweight='bold', va='top')

axs[0, 1].set_xscale('log')
axs[0, 1].grid(True)
axs[0, 1].set_title("Recall")
axs[0, 1].set_xlabel("Threshold")
axs[0, 1].plot(data1[:, 0], data1[:, 3], 'o--', markersize = 3, linewidth = 1, label='Trace reconstruction of zero-discord states')
axs[0, 1].plot(data1[:, 0], data1[:, 6], 'x--', markersize = 4, linewidth = 1, label='Trace reconstruction of separable states')
axs[0, 1].plot(data2[:, 0], data2[:, 3], 'o--', markersize = 3, linewidth = 1, label='Separator reconstruction of zero-discord states')
axs[0, 1].plot(data2[:, 0], data2[:, 6], 'x--', markersize = 4, linewidth = 1, label='Separator reconstruction of separable states')
# Mark subplot with letter
axs[0, 1].text(0.05, 0.9, '(b)', transform=axs[0, 1].transAxes, fontsize=14, fontweight='bold', va='top')

axs[1, 0].grid(True)
axs[1, 0].set_title("Precision vs recall curve")
axs[1, 0].set_ylabel("Precision")
axs[1, 0].set_xlabel("Recall")
axs[1, 0].plot(data1[:, 3], data1[:, 2], 'o--', markersize = 3, linewidth = 1, label='Trace reconstruction of zero-discord states')
axs[1, 0].plot(data1[:, 6], data1[:, 5], 'x--', markersize = 4, linewidth =1,  label='Trace reconstruction of separable states')
axs[1, 0].plot(data2[:, 3], data2[:, 2], 'o--', markersize = 3, linewidth = 1, label='Separator reconstruction of zero-discord states')
axs[1, 0].plot(data2[:, 6], data2[:, 5], 'x--', markersize = 4, linewidth =1,  label='Separator reconstruction of separable states')
# Mark subplot with letter
axs[1, 0].text(0.05, 0.8, '(c)', transform=axs[1, 0].transAxes, fontsize=14, fontweight='bold', va='top')

axs[1, 1].grid(True)
axs[1, 1].set_xscale('log')
axs[1, 1].set_title("F1 score")
axs[1, 1].set_xlabel("Threshold")
axs[1, 1].plot(data1[:, 0], f1_score_zd1, 'o--', markersize = 3, linewidth = 1, label='Trace reconstruction of zero-discord states')
axs[1, 1].plot(data1[:, 0], f1_score_sep1, 'x--', markersize = 4, linewidth =1,  label='Trace reconstruction of separable states')
axs[1, 1].plot(data2[:, 0], f1_score_zd2, 'o--', markersize = 3, linewidth = 1, label='Separator reconstruction of zero-discord states')
axs[1, 1].plot(data2[:, 0], f1_score_sep2, 'x--', markersize = 4, linewidth =1,  label='Separator reconstruction of separable states')
# Mark subplot with letter
axs[1, 1].text(0.05, 0.9, '(d)', transform=axs[1, 1].transAxes, fontsize=14, fontweight='bold', va='top')

axs[2, 0].grid(True)
axs[2, 0].set_xscale('log')
axs[2, 0].set_ylim(0, 1.)
axs[2, 0].set_yticks(np.arange(0, 1.1, 0.5))
axs[2, 0].set_title("Jaccard Similarity")
axs[2, 0].set_xlabel("Threshold")
axs[2, 0].plot(data1[:, 0], data1[:, 1], 'o--', markersize = 3, linewidth = 1, label='Trace reconstruction of zero-discord states')
axs[2, 0].plot(data1[:, 0], data1[:, 4], 'x--', markersize = 4, linewidth =1,  label='Trace reconstruction of separable states')
axs[2, 0].plot(data2[:, 0], data2[:, 1], 'o--', markersize = 3, linewidth = 1, label='Separator reconstruction of zero-discord states')
axs[2, 0].plot(data2[:, 0], data2[:, 4], 'x--', markersize = 4, linewidth =1,  label='Separator reconstruction of separable states')
# Mark subplot with letter
axs[2, 0].text(0.05, 0.9, '(e)', transform=axs[2, 0].transAxes, fontsize=14, fontweight='bold', va='top')

axs[2, 1].grid(True)
axs[2, 1].set_xscale('log')
axs[2, 1].set_ylim(0, 1.)
axs[2, 1].set_yticks(np.arange(0, 1.1, 0.5))
axs[2, 1].set_title("Balanced Accuracy")
axs[2, 1].set_xlabel("Threshold")
axs[2, 1].plot(data1[:, 0], data1[:, 7], 'o--', markersize = 3, linewidth = 1, label='Trace reconstruction of zero-discord states')
axs[2, 1].plot(data1[:, 0], data1[:, 8], 'x--', markersize = 4, linewidth =1,  label='Trace reconstruction of separable states')
axs[2, 1].plot(data2[:, 0], data2[:, 7], 'o--', markersize = 3, linewidth = 1, label='Separator reconstruction of zero-discord states')
axs[2, 1].plot(data2[:, 0], data2[:, 8], 'x--', markersize = 4, linewidth =1,  label='Separator reconstruction of separable states')
# Mark subplot with letter
axs[2, 1].text(0.05, 0.9, '(f)', transform=axs[2, 1].transAxes, fontsize=14, fontweight='bold', va='top')


lbls = axs[1, 0].get_legend_handles_labels()
lgd = fig.legend(*lbls, bbox_to_anchor = (0.65, 0.01), fontsize = 14)
plt.tight_layout(pad = 0.7)
plt.savefig('./plots/all_sep_fc4_mixed_bal.png', bbox_extra_artists=[lgd], bbox_inches='tight', format='png')
plt.close()
