from sklearn.svm import SVC
import numpy as np
np.set_printoptions(suppress=True)

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
            cols_zero.append(line_data[0].split(','))
            cols_one.append(line_data[1].split(','))
            cols_two.append(line_data[2].split(','))
        db = np.r_[np.array(cols_zero), np.array(cols_one), np.array(cols_two)]
        db = db.astype(np.float32)

        return db


def SVM(train_data, test_data):
    '''
    :param train_data: 训练集
    :param test_data: 测试集
    :return: 无
    '''
    # 数据处理
    # np.random.seed(2)
    # np.random.shuffle(test_data)
    # np.random.shuffle(train_data)
    train_fe = train_data[:, :-1]   # 训练特征
    test_fe = test_data[:, :-1]     # 测试特征
    train_le = train_data[:, -1]    # 训练标签
    test_le = test_data[:, -1]      # 测试标签
    # z-score
    temp = np.r_[train_fe, test_fe]
    mu = np.mean(temp, axis=0)
    sigma = np.std(temp, axis=0)
    train_fe = (train_fe - mu) / sigma
    test_fe = (test_fe - mu) / sigma

    # 建模，通过fit计算出对应的决策边界
    clf = SVC(kernel="linear", probability=True, decision_function_shape='ovo').fit(train_fe, train_le)

    z = clf.decision_function(test_fe)  # 输出每个样本分别到三个超平面的距离
    print('每个测试样本点到三个超平面的距离分别为：')
    print(np.around(z, decimals=4))

    pred_prob = clf.predict_proba(test_fe)   # 预测输出的属于哪一类的概率
    pred = clf.predict(test_fe)     # 预测输出所属的类别
    print('每个测试样本点所属类的概率分别为：')
    print('  class1 class2 class7')
    print(np.around(pred_prob, decimals=4))
    print('预测标签：')
    print(pred)
    print('实际标签：')
    print(test_le)  # 真实标签

    acc = clf.score(test_fe, test_le) * 100
    print(f'准确率：{acc}%')
    return acc



if __name__ == '__main__':
    train_db = data_read(r'.\data\SVM\train.txt')
    test_db = data_read(r'.\data\SVM\test.txt')
    acc_SVM = SVM(train_db, test_db)
    acc = np.loadtxt(r'.\data\acc.csv')
    np.savetxt(r'.\data\acc.csv', np.r_[acc, acc_SVM])
