import urllib.request
import os
import tarfile
import numpy as np
import pickle as p
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
tf.reset_default_graph()
from time import time
#下载
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
filepath = 'data/cifar-10-python.tar.gz'
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('downloaded:', result)
else:
    print('Data file already exists.')

#解压
if not  os.path.exists("data/cifar-10-batches-py"):
    tfile = tarfile.open("data/cifar-10-python.tar.gz", 'r:gz')
    result = tfile.extractall('data/')
    print('Extracted to ./data/cifar-10-batches-py/')
else:
    print('Dictionary already exists.')

#导入CIFAR数据集
def load_CIFAR_batch(filename):
    """load single batch of cifar"""
    with open(filename, 'rb') as f:
        #一个样本由标签和图像数据组成
        #3072 = 32 x 32 x 3
        data_dict = p.load(f, encoding= 'bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        #把原始数据结构调整为BCWH batches, channels, width, height
        images = images.reshape(10000, 3, 32, 32)
        #tensorflow 处理图像数据的结构：BWHC
        #把C移动到最后一个维度
        images = images.transpose(0, 2, 3, 1)

        labels = np.array(labels)
        return  images, labels

def load_CIFAR_data(data_dir):
    """load CIFAR data"""
    images_train = []
    labels_train = []
    for i in range(5):
        f = os.path.join(data_dir, 'data_batch_%d' %(i+1))
        print('loading ', f)
        #调用load_CIFAR_batch()获取批量的图像及其对应的标签
        image_batch, label_batch = load_CIFAR_batch(f)
        images_train.append(image_batch)
        labels_train.append(label_batch)
        Xtrain = np.concatenate(images_train)
        Ytrain = np.concatenate(labels_train)
        del image_batch, label_batch
    Xtest, Ytest = load_CIFAR_batch(os.path.join(data_dir, 'test_batch'))
    print('finished loading CIFAR-10 data')
    #返回训练集和测试集的图像和标签
    return Xtrain, Ytrain, Xtest, Ytest
data_dir = 'data/cifar-10-batches-py/'
Xtrain, Ytrain, Xtest, Ytest = load_CIFAR_data(data_dir)
#显示数据集信息
print('training data shape:', Xtrain.shape)
print('training labels shape:', Ytrain.shape)
print('test data shape:', Xtest.shape)
print('test labels shape:', Ytest.shape)
#查看单项image 和 label
#plt.imshow(Xtrain[0])
#print(Ytrain[0])
#plt.show()
#定义标签字典，每一个数字所代表的图像类别名称
label_dict = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer",
              5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"trunk"}
#查看多项images 与 labels
def plot_images_labels_prediction(images,
                                  labels,
                                  prediction,
                                  idx,
                                  num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    if num > 10 :
        num = 10
    for i in range(0, num):
        ax = plt.subplot(2, 5, i+1)
        ax.imshow(images[idx], cmap='binary')
        title = str(i) + ',' +label_dict[labels[idx]]
        if len(prediction) > 0 :
            title += '=>' + label_dict[prediction[idx]]
        ax.set_title(title, fontsize = 10)
        idx += 1
    plt.show()
#先是图像数据及对应标签
#plot_images_labels_prediction(Xtest, Ytest, [], 1, 10)
#数据预处理
#将图像进行数字标准化
Xtrain_normalize = Xtrain.astype('float32') / 255.0
Xtest_normalize = Xtest.astype('float32') / 255.0
encoder = OneHotEncoder(sparse= False)
#标签数据转为one-hot编码
yy = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
encoder.fit(yy)
Ytrain_reshape = Ytrain.reshape(-1, 1)
Ytrain_onehot = encoder.transform(Ytrain_reshape)
Ytest_reshape = Ytrain.reshape(-1, 1)
Ytest_onehot = encoder.transform(Ytest_reshape)
#定义共享函数
#定义权值
def weight(shape):
    #使用截断的正态分布 生成标准差为0.1的随机数来初始化权值
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')
#定义偏置
#初始化为0.1
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')
#定义卷积操作
#步长为1，padding 为same（填充）valid(不填充）
def conv2d(x, W):
    #tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
    #strides为长度为4的一阶张量，并且要求strides[0]=strides[3]=1，
    # strides[1]，strides[2]决定卷积核在输入图像in_hight，in_width方向的滑动步长，
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#定义池化操作
#步长为2，即原尺寸的长和宽各除以2
def max_pool_2x2(x):
    #tf.nn.max_pool(value, ksize, strides, padding, data_format="NHWC", name=None)
    #ksize:池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
    # 因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

#定义网络结构
#输入层
#32*32图像，通道为3(RGB)
with tf.name_scope('input_layer'):
    x = tf.placeholder('float', shape= [None, 32, 32, 3], name="x")

#第1个卷积层
#输入通道：3，输出通道：32，卷积后图像尺寸不变，依然是32x32
with tf.name_scope('conv_1'):
    W1 = weight([3, 3, 3, 32])#[k_width, k_height, input_chn, output_chn]
    b1 = bias([32]) #与output_chn一致
    conv_1 = conv2d(x, W1) + b1
    conv_1 = tf.nn.relu(conv_1)



#第1个池化层
#将32*32图像缩小为16*16，池化不改变通道数量，因此依然是32个
with tf.name_scope('pool_1'):
    pool_1 = max_pool_2x2(conv_1)

#第2个卷积层
#输入通道：32，输出通道：64，卷积后图像尺寸不变，依然是16x16
with tf.name_scope('conv_2'):
    W2 = weight([3, 3, 32, 64])
    b2 = bias([64])
    conv_2 = conv2d(pool_1, W2) + b2
    conv_2 = tf.nn.relu(conv_2)



#第2个池化层
#将16*16图像缩小为8*8，池化不改变通道数量，因此依然是64个
with tf.name_scope('pool_2'):
    pool_2 = max_pool_2x2(conv_2)

#全连接层
#将第二个池化层的64个8*8的图像转化为一维向量，长度是64*8*8 = 4096
#128个神经元
with tf.name_scope('fc'):
    W3 = weight([4096, 128])
    b3 = bias([128])
    flat = tf.reshape(pool_2, [-1, 4096])
    h = tf.nn.relu(tf.matmul(flat, W3) + b3)
    h_dropout = tf.nn.dropout(h, keep_prob = 0.8)

#输出层
#输出层共有10个神经元，对应0-9这10个类别
with tf.name_scope('output_layer'):
    W4 = weight([128, 10])
    b4 = bias([10])
    pred = tf.nn.softmax(tf.matmul(h_dropout, W4) + b4)

#构建模型
with tf.name_scope("optimizer"):
    y = tf.placeholder("float", shape= [None, 10],
                       name="label")
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
    #选择优化器
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss_function)

#定义准确率
with tf.name_scope("evaluation"):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

train_epochs = 25
batch_size = 50
total_batch = int(len(Xtrain)/batch_size)
epoch_list = []
accuracy_list = []
loss_list = []

epoch = tf.Variable(0, name='epoch', trainable=False)

startTime = time()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#断点续训
#设置检查点存储目录
ckpt_dir = "CIFAR_log/"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
#生成saver,max_to_keep 参数，这个是用来设置保存模型的个数，默认为5
saver = tf.train.Saver(max_to_keep= 1)
#如果有检查点文件，读取最新的检查点文件，恢复各种变量值
ckpt = tf.train.latest_checkpoint(ckpt_dir)
if ckpt != None:
    saver.restore(sess, ckpt) #加载所有的参数

    #从这里开始就可以直接使用模型进行预测，或者接着继续预测了  
else:
    print("Training from scratch.")
#获取续训参数
start = sess.run(epoch)
if start >= 24:
    start = 0
print("Training starts from {} epoch.".format(start + 1))
#迭代训练
def get_train_batch(number, batch_size):
    return Xtrain_normalize[number*batch_size:(number + 1)*batch_size],\
                   Ytrain_onehot[number*batch_size:(number + 1)*batch_size]

for ep in range(start, train_epochs):
    for i in range(total_batch):
         batch_x, batch_y = get_train_batch(i, batch_size)
         sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
         if i % 100 == 0:
             print("Step{}".format(i), "finished")
    loss, acc = sess.run([loss_function, accuracy], feed_dict={x: batch_x, y: batch_y})
    epoch_list.append(ep + 1)
    loss_list.append(loss)
    accuracy_list.append(acc)

    print("Train epoch:", '%02d' %(sess.run(epoch) + 1), "Loss = ",'{:.6f}'.format(loss), "Accuracy = ", acc)
    #保存检查点
    saver.save(sess, ckpt_dir + "CIFAR10_cnn_model.cpkt", global_step = ep + 1)
    sess.run(epoch.assign(ep + 1))
duration = time() - startTime
print("Train finished takes:", duration)

#可视化损失值
fig = plt.gcf()
fig.set_size_inches(4, 2)
plt.plot(epoch_list, loss_list, label = 'loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc = 'upper right')
plt.show()
#可视化准确率
plt.plot(epoch_list, accuracy_list, label = 'accuracy')
fig = plt.gcf()
fig.set_size_inches(4, 2)
plt.ylim(0.1, 1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

#评估模型及预测
test_total_batch = int(len(Xtest_normalize)/batch_size)
test_acc_sum = 0.0
for i in range(test_total_batch):
    test_image_batch = Xtest_normalize[i*batch_size:(i+1)*batch_size]
    test_label_batch = Ytest_onehot[i*batch_size:(i+1)*batch_size]
    test_batch_acc = sess.run(accuracy, feed_dict={x:test_image_batch, y:test_label_batch})
    test_acc_sum += test_batch_acc
test_acc = float(test_acc_sum/test_total_batch)
print("Test accuracy:{:.6f}".format(test_acc))

test_pred = sess.run(pred, feed_dict={x:Xtest_normalize[:10]})
prediction_result = sess.run(tf.argmax(test_pred, 1))

plot_images_labels_prediction(Xtest, Ytest, prediction_result, 0, 10)
plt.show()



# ----------------------------------各个层特征可视化-------------------------------
# imput image
fig2, ax2 = plt.subplots(figsize=(2, 2))
ax2.imshow(np.reshape(Xtrain[0], (32, 32, 3)))
plt.show()

# 第一层的卷积输出的特征图
input_image = np.reshape(Xtrain[0], (1, 32, 32, 3))
conv1_16 = sess.run(conv_1, feed_dict={x: input_image})
conv1_transpose = sess.run(tf.transpose(conv1_16, [3, 0, 1, 2]))
print(conv1_transpose.shape) #32x1x32x32
fig3, ax3 = plt.subplots(nrows=4, ncols=8, figsize=(8, 4))

for i in range(4):
    for j in range(8):
        ax3[i][j].imshow(conv1_transpose[i*4+j][0])  # tensor的切片[row, column]


plt.show()

# 第一层池化后的特征图
pool1_16 = sess.run(pool_1, feed_dict={x: input_image})
pool1_transpose = sess.run(tf.transpose(pool1_16, [3, 0, 1, 2]))
fig4, ax4 = plt.subplots(nrows=4, ncols=8, figsize=(8, 4))
print(pool1_transpose.shape) #32x1x16x16
#plt.title('Pool2 32x16x16')
for i in range(4):
    for j in range(8):
        ax4[i][j].imshow(pool1_transpose[i*4+j][0])

plt.show()

# 第二层卷积输出特征图
conv2_32 = sess.run(conv_2, feed_dict={x: input_image})
conv2_transpose = sess.run(tf.transpose(conv2_32, [3, 0, 1, 2]))
fig5, ax5 = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
print(conv2_transpose.shape) #64x1x16x16
#plt.title('Conv2 64x16x16')
for i in range(8):
    for j in range(8):
        ax5[i][j].imshow(conv2_transpose[i*8+j][0])

plt.show()

# 第二层池化后的特征图
pool2_32 = sess.run(pool_2, feed_dict={x: input_image})
pool2_transpose = sess.run(tf.transpose(pool2_32, [3, 0, 1, 2]))
fig6, ax6 = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
print(pool2_transpose.shape)  #64x1x8x8
#plt.title('Pool2 64x8x8')
for i in range(8):
    for j in range(8):
        ax6[i][j].imshow(pool2_transpose[i*8+j][0])

plt.show()
