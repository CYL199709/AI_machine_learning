# 要先运行前两个程序
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 读取数据
train = pd.read_csv('train.txt', sep='\s+', header=None, lineterminator='\n')
test = pd.read_csv('test.txt', sep='\s+', header=None, lineterminator='\n')

# # 训练数据,及类型转换
train = pd.concat([train[0].str.split(',', expand=True).apply(pd.to_numeric),
                   train[1].str.split(',', expand=True).apply(pd.to_numeric),
                   train[2].str.split(',', expand=True).apply(pd.to_numeric)], ignore_index=True, axis=0)
# 测试数据,及类型转换
temp = pd.Series()
for line in test.loc[:, 0]:
    if line != '\r':
        temp = pd.concat([temp, pd.Series(line)], ignore_index=True, axis=0)
for line in test.loc[:, 1]:
    if line != '\r' and line !='NaN':
        temp = pd.concat([temp, pd.Series(line)], ignore_index=True, axis=0)
test = temp.dropna(axis=0).str.split(',', expand=True).apply(pd.to_numeric)
train_X = train.iloc[:, :-1]
train_y = train.iloc[:, -1]
test_X = test.iloc[:, :-1]
test_y = test.iloc[:, -1]

# Z-score
both = pd.concat([train_X, test_X], ignore_index=True, axis=0)
junzhi = both.mean()
bzc = both.std()
train_X = (train_X - junzhi) / bzc
test_X = (test_X - junzhi) / bzc

model = svm.SVC(kernel='linear')
model = model.fit(train_X, train_y)
yucezhi = model.predict(test_X)

zhengquenum = 0
zongshu = len(test_y)
for i in range(zongshu):
    if yucezhi[i] == test_y[i]:
        print('第%d个样本分类正确，类别是%d' %(i, yucezhi[i]))
        zhengquenum = zhengquenum + 1
zhunquelv = zhengquenum * 100 / zongshu
print('正确率是%.2f%%' %zhunquelv)

# 读取之前的正确率
data = pd.read_csv('E:/PyCharm Community Edition 2020.1/Example_Learn/13/accuracy.csv')
zhunquelv_min = float(data.columns.tolist()[0])
zhunquelv_knn = float(data.loc[0].tolist()[0])
zhunquelv_svm = zhunquelv
data = pd.concat([pd.Series(zhunquelv_min), pd.Series(zhunquelv_min), pd.Series(zhunquelv_min)],
                 ignore_index=True,axis=0)
data.plot.bar()
plt.show()

