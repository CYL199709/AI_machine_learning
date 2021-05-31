import numpy as np

label_dict = {0:'Iris-setosa', 1:'Iris-versicolor', 2:'Iris-virginica'}

def data_read(path):
    '''
    :param path: 数据集路径
    :return: 带标签的数据（最后一列），一行为一个样本(np.array())
    '''
    with open(path) as f:
        data = f.readlines()  # txt中所有字符串读入data，得到的是一个list
        cols_zero = []
        cols_one = []
        cols_two = []
        for line in data:
            line_data = line.split('\t')
            line_data = line_data[0:3]
            cols_zero.append(list(line_data[0].split(',')))
            cols_one.append(list(line_data[1].split(',')))
            cols_two.append(list(line_data[2].split(',')))
        cols_zero = np.array(cols_zero)
        cols_one = np.array(cols_one)
        cols_two = np.array(cols_two)
        db = np.r_[cols_zero, cols_one, cols_two]
        # 类别编码
        myindex = np.char.count(db[:,-1], 'setosa') != 0
        myindex = np.where(myindex==True)
        db[myindex, -1] = '0'

        myindex = np.char.count(db[:, -1], 'versicolor') != 0
        myindex = np.where(myindex == True)
        db[myindex, -1] = '1'

        myindex = np.char.count(db[:, -1], 'virginica') != 0
        myindex = np.where(myindex == True)
        db[myindex, -1] = '2'
        # 数据类型转换
        db = db.astype(np.float32)
    return db


def min_dis(train_data, test_data):
    '''
    :param train_data: 训练集
    :param test_data: 测试集
    :return: 无
    '''
    train_fe = train_data[:, :-1]   # 训练特征
    test_fe = test_data[:, :-1]    # 测试特征
    train_le = train_data[:, -1]    # 训练集标签
    test_le = test_data[:, -1]     # 测试集标签
    # z-score
    temp = np.r_[train_fe, test_fe]
    mu = np.mean(temp, axis=0)
    sigma = np.std(temp, axis=0)
    train_fe = (train_fe - mu) / sigma
    test_fe = (test_fe - mu) / sigma
    # 计算类别中心:顺序依次是类别0——类别1——类别2
    class_mean = np.array([])
    for i in range(3):
        class_mean = np.r_[class_mean, np.mean(train_fe[np.where(train_le == i)], axis=0)]
    class_mean = class_mean.reshape([-1, 4])

    # 预测
    right_num = 0
    total_num = len(test_le)
    for num, test in enumerate(test_fe):
        dist = np.sqrt(np.sum(np.square(class_mean - test), axis=1))  # 计算样本与训练集数据的距离
        min_index = np.argsort(dist)
        pred_le = min_index[0]
        if pred_le == test_le[num]:     # 预测正确
            right_num += 1      # 记录正确个数
            print(f'测试集中的第{num}个样本预测正确，类别编号为:{int(pred_le)}——{label_dict[pred_le]}')
    acc = right_num * 100 / total_num
    print(f'准确率：acc = {acc}%')
    return acc


if __name__ == '__main__':
    train_db = data_read(r'.\data\Min_dis\train.txt')
    test_db = data_read(r'.\data\Min_dis\test.txt')
    acc_mindis = min_dis(train_db, test_db)
    np.savetxt(r'.\data\acc.csv', np.array([acc_mindis]))