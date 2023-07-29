import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline

data = np.loadtxt('./results/discord/confusion_acc_std_fancy_train_mixed_sep_test_mixed_avg17.txt', delimiter='  ', skiprows=1)

spl = make_interp_spline(data[::4, 0], data[::4, 2], k=3)
smooth = spl(data[:, 0])

left_0 = np.power(10., -3.85)
left_1 = np.power(10., -2.7)
left_2 = np.power(10., -2.23)
right_0 = np.power(10., -2.85)
right_1 = np.power(10., -2)
right_2 = np.power(10., -1.6)  # -2.4

#x1 = np.logspace(-4, middle_pow, 50)
#x2 = np.logspace(middle_pow, -1, 50)
#x = np.concatenate((x1, x2))
x = data[:, 0]
left_x = np.array([d for d in x if d <= left_2])
left_x1 = np.array([d for d in data[:, 0] if d <= left_1])
left_x2 = np.array([d for d in data[:, 0] if d > left_1 and d <= left_2])
right_x = np.array([d for d in x if d > right_0])
right_x1 = np.array([d for d in data[:, 0] if d > right_0 and d <= right_1])
right_x2 = np.array([d for d in data[:, 0] if d > right_1])

left_a1 = 50./(left_0-left_1)
left_a2 = 50./(left_2-left_1)
left_b1 = 100.-left_a1*left_0
left_b2 = 100.-left_a2*left_2

right_a1 = 50./(right_0-right_1)
right_a2 = 50./(right_2-right_1)
right_b1 = 100.-right_a1*right_0
right_b2 = 100.-right_a2*right_2

left_y1 = left_a1*left_x1 + left_b1
left_y2 = left_a2*left_x2 + left_b2

right_y1 = right_a1*right_x1 + right_b1
right_y2 = right_a2*right_x2 + right_b2

left_y = np.concatenate((left_y1, left_y2))
right_y = np.concatenate((right_y1, right_y2))

plt.grid(True)
plt.xlabel("Threshold")
plt.ylabel('Accuracy [%]')
plt.xscale('log')
#plt.ylim(50, 105)
plt.plot(data[:, 0], data[:, 2], 'o', label='original 15-ensemble')
plt.plot(data[:, 0], smooth, '-', label='smoothing')
plt.plot(left_x, left_y, 'r--', label='th_min1')
plt.plot(right_x, right_y, 'b--', label='th_min2')
plt.plot(left_x, smooth[:len(left_x)]-left_y+50, 'm--', label='smooth $-$ th_min1 + 50')
plt.plot(right_x, smooth[-len(right_x):]-right_y+50, 'g--', label='smooth $-$ th_min2 + 50')
plt.ylim(50,100)
plt.legend()
plt.savefig('./plots/confusion.png', dpi=300)
plt.close()
