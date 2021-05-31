import numpy as np
def normalization(C):
    for i in range(0,4):
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

    C = np.zeros((12*3, 5))
    for i in range(0,12*3):
        B.append(A[i].split(','))

        for j in range(0,4):
            C[i][j] = float(B[i][j])
        if B[i][4] == 'Iris-setosa':
            C[i][4] = 1
        elif B[i][4] == 'Iris-versicolor':
            C[i][4] = 2
        elif B[i][4] == 'Iris-virginica':
            C[i][4] = 3

    C = normalization(C)

    return C

def read_test_data(path):
    A = []
    B = []
    with open(path) as f1:
        for line in f1.readlines():
            for data1 in line.strip('\n').split('\t'):
                A.append(data1)

    C = np.zeros((10 * 3, 5))
    for i in range(0, 10 * 3):
        B.append(A[i].split(','))

        for j in range(0, 4):
            C[i][j] = float(B[i][j])
        if B[i][4] == 'Iris-setosa':
            C[i][4] = 1
        elif B[i][4] == 'Iris-versicolor':
            C[i][4] = 2
        elif B[i][4] == 'Iris-virginica':
            C[i][4] = 3
    C = normalization(C)
    return C

def cal_mean(C):
    M = np.zeros((3,5))
    for i in range(0,len(C)):
        if i % 3 == 0:
            M[0] = M[0] + C[i]
        elif i % 3 == 1:
            M[1] = M[1] + C[i]
        elif i % 3 == 2:
            M[2] = M[2] + C[i]
    M = M/(len(C)/3)
    return M


def classify():
    path = '最小距离\\5\\train.txt'
    C = read_train_data(path)
    M = cal_mean(C)

    return M

def cal_distance(A,B):
    distance = 0
    for i in range(0,len(A)-1):
        distance = distance + (A[i]-B[i])*(A[i]-B[i])
    return distance


def predict(A,M):

    distance1 = cal_distance(A, M[0])
    distance2 = cal_distance(A, M[1])
    distance3 = cal_distance(A, M[2])
    if distance1 < distance2 and distance1 < distance3:
        return 1
    elif distance2 < distance1 and distance2 < distance3:
        return 2
    return 3

def test(M):
    path = '最小距离\\5\\test.txt'
    C = read_test_data(path)


    count = 0
    for i in range(0,len(C)):
        result = predict(C[i], M)
        if result == C[i][4]:
            print("第",i,"个，正确")
            count = count +1
        else:
            print(result,' ',C[i][4])
            print("第", i, "个，错误")
    acc = count / (10*3)
    print("准确率：",acc)

if __name__ == '__main__':

    M = classify()
    test(M)