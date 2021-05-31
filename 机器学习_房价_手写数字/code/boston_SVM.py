# 导入库
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv("data/boston.csv", header = 0)
#显示数据摘要描述信息
print(df.describe())

#获取df的值
df = df.values
#把df转换为np的数组格式
df = np.array(df)

data = df[:, :12]
target = df[:, 12]

# 数据预处理
train_data,test_data,train_target,test_target = train_test_split(data,target,test_size=0.3)
Stand_X = StandardScaler()  # 特征进行标准化
Stand_Y = StandardScaler()  # 标签也是数值，也需要进行标准化
train_data = Stand_X.fit_transform(train_data)
test_data = Stand_X.transform(test_data)
train_target = Stand_Y.fit_transform(train_target.reshape(-1,1)) # reshape(-1,1)指将它转化为1列，行自动确定
test_target = Stand_Y.transform(test_target.reshape(-1,1))
 # ① 线性核函数
clf = LinearSVR(C=2)
clf.fit(train_data,train_target)
y_pred = clf.predict(test_data)
print("线性核函数：")
print("训练集评分：", clf.score(train_data,train_target))
print("测试集评分：", clf.score(test_data,test_target))
print("测试集均方差：",metrics.mean_squared_error(test_target,y_pred.reshape(-1,1)))
print("测试集R2分：",metrics.r2_score(test_target,y_pred.reshape(-1,1)))

# ② 高斯核函数
clf = SVR(kernel='rbf',C=10,gamma=0.1,coef0=0.1)
clf.fit(train_data,train_target)
y_pred = clf.predict(test_data)
print("高斯核函数：")
print("训练集评分：", clf.score(train_data,train_target))
print("测试集评分：", clf.score(test_data,test_target))
print("测试集均方差：",metrics.mean_squared_error(test_target,y_pred.reshape(-1,1)))
print("测试集R2分：",metrics.r2_score(test_target,y_pred.reshape(-1,1)))

# ③ sigmoid核函数
clf = SVR(kernel='sigmoid',C=2)
clf.fit(train_data,train_target)
y_pred = clf.predict(test_data)
print("sigmoid核函数：")
print("训练集评分：", clf.score(train_data,train_target))
print("测试集评分：", clf.score(test_data,test_target))
print("测试集均方差：",metrics.mean_squared_error(test_target,y_pred.reshape(-1,1)))
print("测试集R2分：",metrics.r2_score(test_target,y_pred.reshape(-1,1)))

# ④ 多项式核函数
clf = SVR(kernel='poly',C=2)
clf.fit(train_data,train_target)
y_pred = clf.predict(test_data)
print("多项式核函数：")
print("训练集评分：", clf.score(train_data,train_target))
print("测试集评分：", clf.score(test_data,test_target))
print("测试集均方差：",metrics.mean_squared_error(test_target,y_pred.reshape(-1,1)))
print("测试集R2分：",metrics.r2_score(test_target,y_pred.reshape(-1,1)))
