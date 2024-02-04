import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
import numpy as np

# data1 = np.loadtxt('./results/discord/trace_prediction_thresh_mixed_sep_param_bl_bal_acc_log.txt', delimiter='  ', skiprows=1)
# data2 = np.loadtxt('./results/discord/prediction_thresh_mixed_sep_param_bl_bal_acc_log.txt', delimiter='  ', skiprows=1)

num_gt_zeros_ent = 43597
num_gt_ones_ent = 21403
num_gt_zeros_disc = 26306
num_gt_ones_disc = 38694

data1 = np.loadtxt('./results/3qbits/discord/trace_prediction_thresh_mixed_bal_bal_acc_log.txt', skiprows=1)
data2 = np.loadtxt('./results/3qbits/discord/sep_pure_sep_param_prediction_thresh_mixed_bal_bal_acc_log.txt', skiprows=1)
data3 = np.loadtxt('./results/3qbits/discord/sep_prod_prediction_thresh_mixed_bal_bal_acc_log.txt', skiprows=1)
data4 = np.loadtxt('./results/3qbits/discord/sep_prod_fc4_prediction_thresh_mixed_bal_bal_acc_log.txt', skiprows=1)
data5 = np.loadtxt('./results/3qbits/discord/sep_zd_prediction_thresh_mixed_bal_bal_acc_log.txt', skiprows=1)
data6 = np.loadtxt('./results/3qbits/discord/sep_zd_fc4prediction_thresh_mixed_bal_bal_acc_log.txt', skiprows=1)
data7 = np.loadtxt('./results/3qbits/discord/sep_all_sep_prediction_thresh_mixed_bal_bal_acc_log.txt', skiprows=1)
data8 = np.loadtxt('./results/3qbits/discord/sep_all_sep_fc4_prediction_thresh_mixed_bal_bal_acc_log.txt', skiprows=1)
data9 = np.loadtxt('./results/3qbits/discord/sep_nps_prediction_thresh_mixed_bal_bal_acc_log.txt', skiprows=1)
data10 = np.loadtxt('./results/3qbits/discord/sep_nps_fc4prediction_thresh_mixed_bal_bal_acc_log.txt', skiprows=1)


data_list = [data1, data8, data7, data10]
auc_zd = []
auc_sep = []

for data in data_list:
    # Correct precision if it is equal to 0
    data[:, 2] = np.where(data[:, 2] == 0, 1., data[:, 2])
    data[:, 5] = np.where(data[:, 5] == 0, 1., data[:, 5])

    # Replace precision and recall for non-correlated states with precision and recall for correlated states
    # Discord
    TP_disc = data[:, 3] * num_gt_zeros_disc
    FN_disc = num_gt_zeros_disc - TP_disc
    FP_disc = TP_disc / data[:, 2] - TP_disc
    TN_disc = num_gt_ones_disc - FP_disc

    data[:, 2] = TN_disc / (TN_disc + FN_disc)
    data[:, 3] = TN_disc / num_gt_ones_disc

    # Entanglement
    TP_ent = data[:, 6] * num_gt_zeros_ent
    FN_ent = num_gt_zeros_ent - TP_ent
    FP_ent = TP_ent / data[:, 5] - TP_ent
    TN_ent = num_gt_ones_ent - FP_ent

    data[:, 5] = TN_ent / (TN_ent + FN_ent)
    data[:, 6] = TN_ent / num_gt_ones_ent

    # Zero-discord states
    data_zd = np.array(sorted(data[:, 2:4], key=lambda x: x[1]))
    auc_zd.append(0)
    for i in range(1, len(data_zd)):
        auc_zd[-1] += (data_zd[i, 1] - data_zd[i-1, 1]) * (data_zd[i, 0] + data_zd[i-1, 0]) / 2

    # Separable states
    data_sep = np.array(sorted(data[:, 5:7], key=lambda x: x[1]))
    auc_sep.append(0)
    for i in range(1, len(data_sep)):
        auc_sep[-1] += (data_sep[i, 1] - data_sep[i-1, 1]) * (data_sep[i, 0] + data_sep[i-1, 0]) / 2


# Set global font size
plt.rcParams.update({'font.size': 14})

cols = ["Precision vs recall curve", "Balanced Accuracy"]
rows = ["Pure\nNo FC4", "Prod\nNo FC4", "Prod", "ZD\nNo FC4", "ZD", "Sep\nNo FC4", "Sep", "NPS\nNo FC4", "NPS"]

# get default matplotlib colors
prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

fig, axs = plt.subplots(len(data_list) - 1, 2, figsize=(10, 7.5))

for i in range(0, len(data_list) - 1):
    data2 = data_list[i + 1]

    # if i == 0:
    #     axs[i, 0].set_title(cols[0])
    #     axs[i, 1].set_title(cols[1])

    axs[i, 0].grid(True)
    axs[i, 0].set_ylabel("Precision")
    if i == len(data_list) - 2:
        axs[i, 0].set_xlabel("Recall")
    axs[i, 0].plot(data1[:, 3], data1[:, 2], '-', c=prop_cycle[0], markersize = 3, linewidth = 1, label='Trace reconstruction of non-discordant states')
    axs[i, 0].plot(data1[:, 6], data1[:, 5], '--', c=prop_cycle[3], markersize = 4, linewidth =1,  label='Trace reconstruction of separable states')
    axs[i, 0].plot(data2[:, 3], data2[:, 2], 'o', c=prop_cycle[0], markersize = 3, linewidth = 1, label='Separator reconstruction of non-discordant states')
    axs[i, 0].plot(data2[:, 6], data2[:, 5], 'x', c=prop_cycle[3], markersize = 4, linewidth =1,  label='Separator reconstruction of separable states')

    axs[i, 1].grid(True)
    axs[i, 1].set_xscale('log')
    axs[i, 1].set_ylim(0.4, 1.)
    axs[i, 1].set_ylabel("Accuracy")
    if i == len(data_list) - 2:
        axs[i, 1].set_xlabel("Threshold")
    axs[i, 1].plot(data1[:, 0], data1[:, 7], '-', c=prop_cycle[0], markersize = 3, linewidth = 1, markevery=3, label='Trace reconstruction of non-discordant states')
    axs[i, 1].plot(data1[:, 0], data1[:, 8], '--', c=prop_cycle[3], markersize = 4, linewidth =1, markevery=3, label='Trace reconstruction of separable states')
    axs[i, 1].plot(data2[:, 0], data2[:, 7], 'o', c=prop_cycle[0], markersize = 3, linewidth = 1, markevery=3, label='Separator reconstruction of non-discordant states')
    axs[i, 1].plot(data2[:, 0], data2[:, 8], 'x', c=prop_cycle[3], markersize = 4, linewidth =1, markevery=3, label='Separator reconstruction of separable states')

    axs[i, 0].text(-0.195, 1, letters[i], transform=axs[i, 0].transAxes, size=20)


pad = 5

# for ax, col in zip(axs[0], cols):
#     ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
#                 xycoords='axes fraction', textcoords='offset points',
#                 size='large', ha='center', va='baseline')

# anns = []
# for ax, row in zip(axs[:,0], rows):
#     anns.append(ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
#                 xycoords=ax.yaxis.label, textcoords='offset points',
#                 size='large', ha='right', va='center'))

# lbls = axs[1, 0].get_legend_handles_labels()
# lgd = fig.legend(*lbls, bbox_to_anchor = (0.95, 0.01), fontsize = 14)
plt.tight_layout(pad = 0.7, rect=[0.1, 0, 1.2, 1])
plt.savefig('./plots/arch_metrics_corr_reduced.png', bbox_inches='tight', format='png')
plt.close()
