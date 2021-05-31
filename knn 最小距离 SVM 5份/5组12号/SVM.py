
import numpy as np
from sklearn import svm





def normalization(C):
    for i in range(1,10):
        var = np.var(C[:,i])
        mean = np.mean(C[:,i])
        C[:,i] = (C[:,i] - mean) / var

    return C

def read_train_data(path):
    A = []
    B = []
    with open(path) as f1:
        for line in f1.readlines():
            for data1 in line.strip('\n').split('\t'):
                A.append(data1)

    C = np.zeros((22*3, 11))
    for i in range(0,22*3):
        B.append(A[i].split(','))

        for j in range(0,11):
            C[i][j] = float(B[i][j])


    C = normalization(C)

    return C

def read_test_data(path):
    A = []
    B = []
    with open(path) as f1:
        for data1 in f1.readlines():
                A.append(data1)

    C = np.zeros((37, 11))
    for i in range(0, 37):
        B.append(A[i].split(','))
        for j in range(0, 11):
            C[i][j] = float(B[i][j])

    C = normalization(C)
    return C




if __name__ == '__main__':

    A = read_train_data("SVM\\5\\train.txt")

    B = read_test_data("SVM\\5\\test.txt")



    classifier = svm.SVC(C=1, kernel='linear')
    classifier.fit(A[:,1:10], A[:,10])  # ravel函数在降维时默认是行序优先

    tra_label = classifier.predict(A[:,1:10])  # 训练集的预测标签
    tes_label = classifier.predict(B[:,1:10])  # 测试集的预测标签

    print(tra_label)
    print(tes_label)
    print("训练集：", classifier.score(A[:,1:10], A[:,10]))
    print("测试集：", classifier.score(B[:,1:10], B[:,10]))



