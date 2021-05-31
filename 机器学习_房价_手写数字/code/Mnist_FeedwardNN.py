# mnist 训练测试demo

import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
import pandas as pd

import pylab
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

'''从mnist数据集读入训练数据'''

mnist = input_data.read_data_sets('data\\MNIST_data\\', one_hot=True)

'''定义模型'''

x = tf.placeholder(tf.float32, [None, 784], name='input_x')  # 图像输入向量
w = tf.Variable(tf.zeros([784, 10]), name='w')  # 权值向量
b = tf.Variable(tf.zeros([10]), name='bias')  # 偏置，初始化值为全零

y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')  # 预测结果
y_ = tf.placeholder("float", [None, 10], name='y_')  # 实际结果

# 计算交叉熵
cross_entropty = -tf.reduce_sum(y_ * tf.log(y))
# cross_entropty = tf.reduce_sum(tf.square(y_ - y)/100)

# 使用梯度下降算法进行反向传播 学习率为 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropty)

# 初始化计算图中的变量
init = tf.global_variables_initializer()

# 建立会话 初始化变量
sess = tf.Session()
sess.run(init)

'''开始训练'''
'''开始训练模型 先训练300次'''
crs_ety = []
acc_ = []
for i in range(300):
    # 每次选取100条图片数据训练
    # batch_xs、batch_ys 分别代表 图片的像素数组 和 标签数组
    batch_xs, batch_ys = mnist.train.next_batch(100)
    result = sess.run([train_step, cross_entropty], feed_dict={x: batch_xs, y_: batch_ys})
    # 输出交叉熵
    print("cross_entropy:%f" % result[1])
    crs_ety.append(result[1])

    # 训练一次之后就计算一下模型的预测准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 运行准确率计算过程 并且输出准确率
    accu = sess.run([accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("accuracy：%.5f" % accu[0])
    acc_.append(accu[0])

test = pd.DataFrame(data=crs_ety)
test.to_csv(r'res.csv')

plt.plot(crs_ety)
plt.xlabel('批次')
plt.ylabel('损失值')
# plt.show()
plt.savefig('FNN crs_ety.png')
plt.text(300, 300, 'cross_entropy')
plt.close()

plt.plot(acc_)
plt.xlabel('批次')
plt.ylabel('准确率')
# plt.show()
plt.savefig('FNN acc_.png')
plt.text(300, 300, 'accuracy')


