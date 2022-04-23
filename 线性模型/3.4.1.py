from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import sklearn.datasets as ds
import pandas as pd
import numpy as np

# 十折交叉验证生成训练集和测试集
def tenfolds(data):
    accuracy=0
    kf = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)
    for x_train_index, x_test_index in kf.split(data['data']):
        x_train = data['data'][x_train_index]
        y_train = data['target'][x_train_index]
        x_test = data['data'][x_test_index]
        y_test = data['target'][x_test_index]
        # 用对率回归进行训练，拟合数据
        model = LogisticRegression(multi_class= 'ovr', solver = 'liblinear')
        model.fit(x_train, y_train)
        # 用训练好的模型预测
        y_pred = model.predict(x_test)
        true=0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                true+=1
        accuracy+=true/len(y_pred)

    print("10折交叉-对率回归（调库）的精度是：", accuracy/len(y_pred))

# 用留一法验证
def leaveone(data):
    loo = model_selection.LeaveOneOut()
    i = 0
    accuracy=0
    for x_train_index, x_test_index in loo.split(data['data']):
        x_train = data['data'][x_train_index]
        y_train = data['target'][x_train_index]
        x_test = data['data'][x_test_index]
        y_test = data['target'][x_test_index]
        # 用对率回归进行训练，拟合数据
        model = LogisticRegression(multi_class='ovr', solver='liblinear')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        if y_pred == y_test:
            accuracy += 1
        i+=1
    # 计算精度
    accuracy = accuracy/100
    print("留一法-对率回归（调库）的精度是：", accuracy)

df = pd.read_csv('iris.data',header=None)
df=df.values.tolist()
x=[]
y=[]
for xy in df:
    x.append(xy[:4])
    if xy[4]=='Iris-setosa':
        y.append(0)
    else:
        y.append(1)
x=np.array(x)
y=np.array(y)
data={'data':x,'target':y}
tenfolds(data)
leaveone(data)