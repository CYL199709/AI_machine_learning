import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 读取数据
train = pd.read_csv('train.txt', sep='\s+', header=None, lineterminator='\n')
test = pd.read_csv('test.txt', sep='\s+', header=None, lineterminator='\n')
# 训练拆分数据类别,及类型转换
train1 = train[0].str.replace('Iris-setosa', '0').str.split(',', expand=True).apply(pd.to_numeric)
train2 = train[1].str.replace('Iris-versicolor', '1').str.split(',', expand=True).apply(pd.to_numeric)
train3 = train[2].str.replace('Iris-virginica', '2').str.split(',', expand=True).apply(pd.to_numeric)
# 测试数据
test = pd.concat(
    [test[0].str.replace('Iris-virginica', '2').str.split(',', expand=True).apply(pd.to_numeric),
     test[1].str.replace('Iris-versicolor', '1').str.split(',', expand=True).apply(pd.to_numeric),
     test[2].str.replace('Iris-setosa', '0').str.split(',', expand=True).apply(pd.to_numeric)
     ], ignore_index=True, axis=0
)

# Z-score
both = pd.concat([train1.loc[:, 0:3], train2.loc[:, 0:3], train3.loc[:, 0:3], test.loc[:, 0:3]],
                 ignore_index=True, axis=0)
junzhi = both.mean()
bzc = both.std()
train1.loc[:, 0:3], train2.loc[:, 0:3], train3.loc[:, 0:3] = \
    (train1.loc[:, 0:3] - junzhi) / bzc, (train2.loc[:, 0:3] - junzhi) / bzc, (train3.loc[:, 0:3] - junzhi) / bzc
test.loc[:, 0:3] = (test.loc[:, 0:3] - junzhi) / bzc

# 计算训练集中的每类的中心
centtrain1 = train1.loc[:, 0:3].mean()
centtrain2 = train2.loc[:, 0:3].mean()
centtrain3 = train3.loc[:, 0:3].mean()
# print(centtrain3)
center = pd.concat([centtrain1, centtrain2, centtrain3], ignore_index=True, axis=1).transpose()

# 准确率计算
zhengquenum = 0
zongshu = len(test)
for line in range(zongshu):
    temp = pd.concat([test.loc[line, 0:3], test.loc[line, 0:3], test.loc[line, 0:3]],
                     ignore_index=True, axis=0)
    juli = np.sqrt(np.sum(np.square(center - temp), axis=1))     # 计算测试样本到中心距离
    # 预测值
    yucezhi = 0
    tempmin = 100000
    for i in range(3):
        if juli[i] <= tempmin:
            tempmin = juli[i]
            yucezhi = i
    if yucezhi == int(test.loc[line:line, 4]):
        zhengquenum = zhengquenum + 1
        print('第%d个样本正确分类，被分到第%d类' %(line, yucezhi))
zhunquelv = zhengquenum * 100 / zongshu
print('正确率是%.2f%%' %zhunquelv)

# 保存正确率
data = pd.Series(zhunquelv)
data.to_csv('E:/PyCharm Community Edition 2020.1/Example_Learn/13/accuracy.csv', header=False, index=False)


