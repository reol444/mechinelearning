import numpy as np
import pandas as pd
from LR import LR
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
one = []
#读入西瓜数据
wm_data=pd.read_excel('watermelon3.0a.xlsx')
# #得到(x;1)
# x=np.array([list(wm_data['密度']),list(wm_data['含糖率']),[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
# #得到y
# y=np.array(wm_data['好瓜'])
# #得到beta=(w;b),初始值设置为(1，1，1)
# beta=np.array([[1],[1],[1]])
# beta_s,beta_list=LR(x,y,beta,10)
# beta_s=np.array(beta_s)
# beta_list=np.array(beta_list).T.tolist()
# iter=np.linspace(1,10,10)
# plt.plot(iter,beta_list[2])
# plt.xlabel("iter")
# plt.ylabel("b")
# plt.title("b-iter")
# plt.show()
# w=[beta_s[0][0],beta_s[1][0]]
# b=beta_s[2][0]
# print("迭代1000次后，w的值为{0}，b的值为{1}。".format(w,b))

def predict(x_test,beta_s):
    pred=[]
    x_test = list(zip(list(x_test.T[0]), list(x_test.T[1]), one))
    x_test = np.array(x_test).T
    betaT_x = np.array(np.dot(beta_s.T[0], x_test))
    p = np.exp(betaT_x) / (1 + np.exp(betaT_x))
    for n in p:
        if n <= 0.5:
            pred.append(0)
        else:
            pred.append(1)
    return pred

def holdout(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    x_train = list(zip(list(x_train.T[0]), list(x_train.T[1]), one))
    x_train = np.array(x_train).T
    beta = np.array([[1], [1], [1]])
    beta_s = np.array(LR(x_train, y_train, beta, 5))
    pred = predict(x_test, beta_s)
    accuracy = np.mean(pred == y_test)
    print("西瓜3.0a上留出法-对率回归：")
    print("真实值为：", list(y_test))
    print("预测值为：", pred)
    print("精度为：", accuracy)
    precision=metrics.precision_score(y_test, pred)
    print("查准率为：", precision)
    recall=metrics.recall_score(y_test, pred)
    print("查全率为：", recall)
    f1 = 2 * precision * recall / (precision + recall)
    print("F1的值是：", f1)

def myleaveone(x,y):
    loo = model_selection.LeaveOneOut()
    accuracy = 0
    beta_list=[]
    for i in range(len(x)):
        one.append(1)
    for x_train_index, x_test_index in loo.split(x):
        x_train = x[x_train_index]
        y_train = y[x_train_index]
        x_test = x[x_test_index]
        y_test = y[x_test_index]
        x_train = list(zip(list(x_train.T[0]), list(x_train.T[1]),one))
        x_train = np.array(x_train).T
        beta = np.array([[1], [1], [1]])
        beta_s = np.array(LR(x_train, y_train, beta, 5))
        beta_list.append(beta_s)
        pred=predict(x_test,beta_s)
        accuracy+=np.mean(pred==y_test)
    # for b in beta_list:
    #     print(b[0],b[1],b[2])
    print("在西瓜3.0a上留一法-对率回归的精度是：", accuracy / len(x))

def mytenfold(x,y):
    kf = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)
    accuracy = 0
    one = []
    beta_list=[]
    for i in range(17):
        one.append(1)
    for x_train_index, x_test_index in kf.split(x):
        x_train = x[x_train_index]
        y_train = y[x_train_index]
        x_test = x[x_test_index]
        y_test = y[x_test_index]
        x_train = list(zip(list(x_train.T[0]), list(x_train.T[1]), one))
        x_train = np.array(x_train).T
        beta = np.array([[1], [1], [1]])
        beta_s = np.array(LR(x_train, y_train, beta, 5))
        beta_list.append(beta_s)
        pred = predict(x_test, beta_s)
        accuracy += np.mean(pred == y_test)
    # for b in beta_list:
    #     print(b[0], b[1], b[2])
    print("在西瓜3.0a上10折交叉-对率回归的精度是：", accuracy / 10)
#得到(x;1)
x=np.array([list(wm_data['密度']),list(wm_data['含糖率'])]).T
#得到y
y=np.array(wm_data['好瓜'])
myleaveone(x,y)
mytenfold(x,y)
holdout(x,y)