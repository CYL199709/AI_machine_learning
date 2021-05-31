
import numpy as np


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler




def normalization(C):
    for i in range(1,14):
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

    C = np.zeros((16*3, 14))
    for i in range(0,16*3):
        B.append(A[i].split(','))

        for j in range(0,14):
            C[i][j] = float(B[i][j])


    C = normalization(C)

    return C

def read_test_data(path):
    A = []
    B = []
    with open(path) as f1:
        for line in f1.readlines():
            for data1 in line.strip('\n').split('\t'):
                A.append(data1)

    C = np.zeros((33 * 3, 14))
    for i in range(0, 33 * 3):
        B.append(A[i].split(','))

        if len(B[i]) == 14:
            for j in range(0, 14):
                C[i][j] = float(B[i][j])

    C = normalization(C)
    return C




if __name__ == '__main__':

    A = read_train_data("近邻\\5\\train.txt")

    B = read_test_data("近邻\\5\\test.txt")

    # n_neighbors代表近邻数，p=2代表欧式距离，p=1代表曼哈顿距离
    # metric='minkowski'代表闵可夫斯基距离，他是对欧氏距离和曼哈顿距离的一种泛化
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto', weights='distance', n_jobs=1)
    knn.fit(A[:,1:14], A[:,0])
    print(knn.predict(B[:,1:14]))
    print(knn.score(B[:,1:14],B[:,0]))
