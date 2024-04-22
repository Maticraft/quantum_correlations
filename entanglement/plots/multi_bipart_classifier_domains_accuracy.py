import sys
sys.path.append('./')

from commons.data.datasets import BipartitionMatricesDataset
from commons.models.separators import FancySeparator, rho_reconstruction

import numpy as np
import matplotlib.pyplot as plt


def load_acc_with_ratio(path):
    with open(path, 'r') as f:
        data = f.readlines()[1:]
    data = [x.strip().split('  ')[1:7] for x in data]
    acc = [float(x) for x in data[0]]
    ratio = [float(x) for x in data[1]]
    return acc, ratio


def plot_filled(ax, x_values, y_values, color):
    ax.plot(x_values, y_values, color=color)
    ax.fill_between(x_values, y_values, alpha=0.3, color=color)


def addlabels2f(ax, x,y, color):
    for i in range(len(x)):
        ax.text(0.95*x[i], y[i] + 0.005, f'{y[i]:.2f}', ha = 'center', color=color, fontsize=22)

def addlabels1f(ax, x,y, color):
    for i in range(len(x)):
        ax.text(0.95*x[i], y[i] + 0.5, f'{y[i]:.1f}', ha = 'center', color=color, fontsize=22)


common_results_path = './results/3qbits/multi_class_siam_eq_log_10/no_pptes_bisep/weights05_ep10_cnn_class_best_val_loss_{}_test.txt'
save_path = './plots/domain_accuracy_{}.png'

logscale = True

thresholds = [1.e-4, 2.e-4, 5.e-4, 1.e-3, 2.e-3, 5.e-3, 1.e-2, 2.e-2, 5.e-2, 1.e-1]
# set x labels to the middle of the intervals
x_labels = [np.log((np.exp(thresholds[i]) + np.exp(thresholds[i+1])) / 2) for i in range(len(thresholds) - 1)]

plots = {
    'Validation dataset': 0,
    'mixed': 2,
    'Acin PPTES': 3,
    'Horodecki PPTES': 4,
    'UPB PPTES': 5
}

plt.rcParams.update({'font.size': 28})

for plot_name, plot_idx in plots.items():
    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax2 = ax1.twinx()    
    plt.title(plot_name)
    if logscale:
        plt.xscale('log')
    accs = []
    ratios = []
    for i in range(10):
        acc, ratio = load_acc_with_ratio(common_results_path.format(i))
        accs.append(acc[plot_idx])
        ratios.append(ratio[plot_idx])

    # plot_filled(ax1, x_labels, accs, color='xkcd:blue')
    ratios = ratios[1:]
    accs = accs[1:]

    widths = [thresholds[i+1] - thresholds[i] for i in range(len(thresholds) - 1)]
    # filter out widths, accs, and x_labels, where accs is 0
    widths_acc = [widths[i] for i in range(len(accs)) if accs[i] >= 0.1 and ratios[i] >= 0.01]
    x_labels_acc = [x_labels[i] for i in range(len(accs)) if accs[i] >= 0.1 and ratios[i] >= 0.01]
    accs = [accs[i] for i in range(len(accs)) if accs[i] >= 0.1 and ratios[i] >= 0.01]
    ax1.bar(x_labels_acc, accs, width=widths_acc, color='xkcd:blue', alpha=0.5, edgecolor='black')
    # ax1.bar_label(ax1.containers[0], fmt='%1.1f', label_type='edge', color='xkcd:dark blue')
    addlabels1f(ax1, x_labels_acc, accs, color='xkcd:dark blue')
    # plot_filled(ax2, x_labels, ratios, color='xkcd:red')
    x_labels_ratio = [x_labels[i] for i in range(len(ratios)) if ratios[i] >= 0.01]
    widths_ratio = [widths[i] for i in range(len(ratios)) if ratios[i] >= 0.01]
    ratios = [ratios[i] for i in range(len(ratios)) if ratios[i] >= 0.01]
    ax2.bar(x_labels_ratio, ratios, width=widths_ratio, color='xkcd:red', alpha=0.5, edgecolor='black')
    # ax2.bar_label(ax2.containers[0], fmt='%1.2f', label_type='edge', color = 'xkcd:dark red')
    addlabels2f(ax2, x_labels_ratio, ratios, color='xkcd:dark red')
    ax1.set_xlabel('Separability measure M')
    # Move x ticks downwards
    ax1.tick_params(axis='x', direction='in', pad=18)

    ax1.set_ylabel('Accuracy', color='xkcd:dark blue')
    ax1.tick_params(axis='y', labelcolor='xkcd:dark blue')
    ax2.set_ylabel('Ratio', color='xkcd:dark red')
    ax2.tick_params(axis='y', labelcolor='xkcd:dark red')
    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 1.05)
    ax1.set_xlim(1.e-4, 0.1)

    # add extra space over the bars

    fig.tight_layout()
    plt.savefig(save_path.format(plot_name))
    plt.close()
