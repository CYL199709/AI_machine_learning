# 运行此段程序要先运行“最小距离.py”
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 读取数据
train = pd.read_csv('train.txt', sep='\s+', header=None, lineterminator='\n')
test = pd.read_csv('test.txt', sep='\s+', header=None, lineterminator='\n')


# 训练数据,及类型转换
train = pd.concat([train[0].str.split(',', expand=True).apply(pd.to_numeric),
                   train[1].str.split(',', expand=True).apply(pd.to_numeric),
                   train[2].str.split(',', expand=True).apply(pd.to_numeric)], ignore_index=True, axis=0)
# 测试数据,及类型转换
test = pd.concat([test[0].str.split(',', expand=True).apply(pd.to_numeric),
                  test[1].str.split(',', expand=True).apply(pd.to_numeric),
                  test[2].str.split(',', expand=True).apply(pd.to_numeric),], ignore_index=True, axis=0)

# Z-score
both = pd.concat([train.loc[:, 1:], test.loc[:, 1:]], ignore_index=True, axis=0)    # 无标签数据
junzhi = both.mean()
bzc = both.std()

train.loc[:, 1:] = (train.loc[:, 1:] - junzhi) / bzc
test.loc[:, 1:] = (test.loc[:, 1:] - junzhi) / bzc

K = 6   # k值
# 计算距离及预测
zhengquenum = 0
zongshu = len(test)
for line in range(zongshu):
    juli = np.sqrt(np.sum(np.square(train.loc[:, 1:] - test.loc[line, 1:]), axis=1))
    julibiaoqian = pd.concat([juli, train[0]], ignore_index=True, axis=1)
    paixu = julibiaoqian.sort_values(0, axis=0, ignore_index=True)     # 排序结果
    yucezhi = paixu.loc[:K-1, 1].value_counts()
    # print(paixu.loc[:K-1, :])
    yucezhi = yucezhi._stat_axis.values[0]
    if int(yucezhi) == int(test.loc[line, 0]):
        zhengquenum = zhengquenum + 1
        print('第%d个样本正确分类，被分到第%d类' % (line, yucezhi))
    # print('#' * 12)
zhunquelv = zhengquenum * 100 / zongshu
print('正确率是%.2f%%' %zhunquelv)

# 保存正确率
data = pd.read_csv('E:/PyCharm Community Edition 2020.1/Example_Learn/13/accuracy.csv').columns.tolist()
data1 = pd.concat([pd.Series(data), pd.Series(zhunquelv)], ignore_index=True, axis=0)
data1.to_csv('E:/PyCharm Community Edition 2020.1/Example_Learn/13/accuracy.csv', header=False, index=False)


