import numpy as np
from PIL import Image

image = Image.open("手写//2.bmp")
t = np.array(image) / 255



testfeature = t.reshape(1,784)


np.set_printoptions(threshold=np.inf, suppress=True)


myweight = np.loadtxt("myweight1.csv", delimiter=",")

testfeatureMatrix = np.append(testfeature, np.ones(shape=(len(testfeature), 1)), axis=1)
mypredict = np.dot(testfeatureMatrix, myweight)
expmpre = np.exp(mypredict)
expsum = np.sum(expmpre, axis=1)

for i in range(len(testfeature)):
    expmpre[i, :] = expmpre[i, :] / expsum[i]
    pre = np.argmax(expmpre[i, :])
    print(pre)

print("-" * 20)



