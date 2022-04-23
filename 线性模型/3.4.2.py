from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import sklearn.datasets as ds
import pandas as pd
import numpy as np
from LR import LR

def mytenfold(x,y):
    kf = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)
    accuracy = 0
    one = []
    for i in range(100):
        one.append(1)
    for x_train_index, x_test_index in kf.split(x):
        x_train = x[x_train_index]
        y_train = y[x_train_index]
        x_test = x[x_test_index]
        y_test = y[x_test_index]
        x_train = list(zip(list(x_train.T[0]), list(x_train.T[1]), list(x_train.T[2]), list(x_train.T[3]), one))
        x_train = np.array(x_train).T
        beta = np.array([[1], [1], [1], [1], [1]])
        beta_s = np.array(LR(x_train, y_train, beta, 10))
        x_test = list(zip(list(x_test.T[0]), list(x_test.T[1]), list(x_test.T[2]), list(x_test.T[3]), one))
        x_test = np.array(x_test).T
        betaT_x = np.array(np.dot(beta_s.T[0], x_test))
        p = np.exp(betaT_x) / (1 + np.exp(betaT_x))
        pred = []
        for n in p:
            if n <= 0.5:
                pred.append(0)
            else:
                pred.append(1)
        true = 0
        for i in range(len(p)):
            if pred[i] == y_test[i]:
                true += 1
        accuracy += true / len(p)
    print("10折交叉-对率回归（自己实现）的精度是：", accuracy / 10)

def myleaveone(x,y):
    loo = model_selection.LeaveOneOut()
    accuracy = 0
    one = []
    for i in range(100):
        one.append(1)
    for x_train_index, x_test_index in loo.split(x):
        x_train = x[x_train_index]
        y_train = y[x_train_index]
        x_test = x[x_test_index]
        y_test = y[x_test_index]
        x_train = list(zip(list(x_train.T[0]), list(x_train.T[1]), list(x_train.T[2]), list(x_train.T[3]), one))
        x_train = np.array(x_train).T
        beta = np.array([[1], [1], [1], [1], [1]])
        beta_s = np.array(LR(x_train, y_train, beta, 10))
        x_test = list(zip(list(x_test.T[0]), list(x_test.T[1]), list(x_test.T[2]), list(x_test.T[3]), one))
        x_test = np.array(x_test).T
        betaT_x = np.array(np.dot(beta_s.T[0], x_test))
        p = np.exp(betaT_x) / (1 + np.exp(betaT_x))
        if p <= 0.5:
            pred=0
        else:
            pred=1
        true=0
        if pred==y_test:
            true = 1
        accuracy += true / len(p)
    print("留一法-对率回归（自己实现）的精度是：", accuracy / len(x))

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
print(x)
x = (x - x.mean(0)) / x.std(0)
y=np.array(y)
mytenfold(x,y)
myleaveone(x,y)