import numpy as np


def data_read(path):
    '''
    :param path: 数据集路径
    :return: 带标签的数据(第一列），一行为一个样本(np.array())
    '''
    with open(path) as f:
        data = f.readlines()  # txt中所有字符串读入data，得到的是一个list

        container = []
        for line in data:
            line_data = line.split('\t')
            for i in range(len(line_data)):
                temp = list(line_data[i].split(','))
                if len(temp) > 1:
                    container.append(temp)
        container = np.array(container)
        db = container.astype(np.float32)
    return db


def kNN(train_data, test_data, k):
    '''
    :param train_data: 训练集
    :param test_data: 测试集
    :return: 无
    '''
    train_fe = train_data[:, 1:]  # 训练特征
    test_fe = test_data[:, 1:]  # 测试特征
    train_le = train_data[:, 0]  # 训练集标签
    test_le = test_data[:, 0]  # 测试集标签
    # z-score
    temp = np.r_[train_fe, test_fe]
    mu = np.mean(temp, axis=0)
    sigma = np.std(temp, axis=0)
    train_fe = (train_fe - mu) / sigma
    test_fe = (test_fe - mu) / sigma

    # 预测
    right_num = 0
    total_num = len(test_le)
    for num, test in enumerate(test_fe):
        dist = np.sqrt(np.sum(np.square(train_fe - test), axis=1))  # 计算样本与训练集数据的距离
        min_index = np.argsort(dist)
        k_lable = train_le[min_index[:k]]   # 取前k个最小距离标签
        y_pred = np.argmax(np.bincount(k_lable.astype(np.int16)))
        if y_pred == test_le[num]:  # 预测正确
            right_num += 1  # 记录正确个数
            print(f'测试集中的第{num}个样本预测正确，类别编号为:{int(y_pred)}')
    acc = right_num * 100 / total_num
    print(f'准确率：acc = {acc}%')
    return acc



if __name__ == '__main__':
    train_db = data_read(r'.\data\KNN\train.txt')
    test_db = data_read(r'.\data\KNN\test.txt')
    acc_KNN = kNN(train_db, test_db, 8)
    acc = np.loadtxt(r'.\data\acc.csv')
    np.savetxt(r'.\data\acc.csv', np.r_[acc, acc_KNN])

