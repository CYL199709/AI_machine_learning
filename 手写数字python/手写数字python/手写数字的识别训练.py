
import numpy as np
np.set_printoptions(threshold=np.inf, suppress=True)

feature = np.loadtxt("train_image.csv", delimiter=",", max_rows=6000) / 255
featureMatrix = np.append(feature, np.ones(shape=(len(feature), 1)), axis=1)

weightm = np.ones(shape=(feature.shape[1], 10))
weightb = np.ones(shape=(1, 10))
weight = np.r_[weightm, weightb]
learningrate = 0.0001

label = np.loadtxt("train_label_hotencoding.csv", delimiter=",", max_rows=6000)


def grandientDecent():
    predict2sigmod = 1/ (1 + np.exp(-np.dot(featureMatrix, weight)))
    temp_slop = np.dot(featureMatrix.T, predict2sigmod - label)
    return temp_slop


def train():
    global weight
    for i in range(1, 50000):
        slopmb = grandientDecent()
        weight -= slopmb * learningrate
    return weight


if __name__ == '__main__':
    myweight = train()
    np.savetxt("myweight1.csv",myweight,  fmt="%f", delimiter=",")
