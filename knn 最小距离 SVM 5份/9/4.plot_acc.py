import numpy as np
import matplotlib.pyplot as plt
# 在执行改程序前要依次执行1.Min_dis.py——>2.kNN.py——>3.SVM.py
acc = np.loadtxt(r'.\data\acc.csv')
x = np.arange(3)

plt.bar(x, acc)
plt.ylabel('accuracy')
ax = plt.gca()
ax.set_xticks(x)
ax.set_xticklabels(('Min_dist', 'KNN', 'SVM'))

for i, value in enumerate(acc):
    plt.text(i, value + 4.5, '%.2f%%' % value,ha='center', va='top')
plt.show()
