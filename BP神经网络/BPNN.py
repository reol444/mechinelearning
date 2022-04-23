import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
seed = 2022
np.random.seed(seed)

#数据预处理
def data_processing(data):
    #将数据处理为标签编码
    for col in data.columns:
        if data[col].dtype == 'object':
            encoder = pp.LabelEncoder()
            data[col] = encoder.fit_transform(data[col])
    #划分出输入和真实值
    x = data.drop('好瓜', axis=1)
    y = data['好瓜']
    #将输入集去均值和均值归一化，便于梯度下降快速收敛和精度提高
    ss = pp.StandardScaler()
    x = ss.fit_transform(x)
    return x, y

#定义sigmoid函数
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

#标准bp
def standard_bp(x,y,q,eta,max_iter):
    #初始化两个输入和阈值
    v = np.random.random((x.shape[1], q))
    gama = np.random.random((1, q))
    w = np.random.random((q, 1))
    sigma = np.random.random((1, 1))
    #损失列表，便于画图
    losslist=[]
    #开始训练
    for ite in range(max_iter):
        loss_per_iter = []
        for i in range(x.shape[0]): #对输入的每一维
            xi = x[i, :].reshape(1,x[i, :].shape[0]) #(1, x.shape[1])
            yi = y[i, :].reshape(1,y[i, :].shape[0]) #(1,1)
            #计算出y_pred等
            alpha=np.dot(xi,v) #(1, q)
            b=sigmoid(alpha-gama) #(1, q)
            beta=np.dot(b,w) #(1,1)
            y_pred = sigmoid(beta-sigma) #(1,1)
            #计算当前的误差
            loss = np.square(y_pred - yi)/2
            loss_per_iter.append(loss)
            #反向传播
            g=y_pred*(1-y_pred)*(yi-y_pred) #(1,1)
            e=b*(1-b)*np.dot(g,np.transpose(w)) #(1, q)
            #更新
            v = v + eta * np.dot(np.transpose(xi),e) #(x.shape[1],q)
            w = w + eta * np.dot(np.transpose(b),g) #(q,1)
            gama = gama - eta * e #(1,q)
            sigma = sigma - eta * g #(1,1)
        #取每次更新损失的平均值作为一次迭代的损失
        losslist.append(np.mean(loss_per_iter))
    return v, w, gama, sigma, losslist

def cumulative_bp(x,y,q,eta,max_iter):
    # 初始化两个输入和阈值
    v = np.random.random((x.shape[1], q))
    gama = np.random.random((1, q))
    w = np.random.random((q, 1))
    sigma = np.random.random((1, 1))
    # 损失列表，便于画图
    losslist = []
    # 开始训练
    for ite in range(max_iter):
        #计算出y_pred等
        alpha = np.dot(x, v) #(x.shape[0], q)
        b = sigmoid(alpha - gama) #(x.shape[0], q)
        beta = np.dot(b, w) #(x.shape[0],1)
        y_pred = sigmoid(beta - sigma) #(x.shape[0],1)
        # 计算误差，加入列表便于画图
        loss = np.mean(np.square(y - y_pred))/2
        losslist.append(loss)
        # 反向传播
        g = y_pred * (1 - y_pred) * (y - y_pred) #(x.shape[0],1)
        e = b * (1 - b) * np.dot(g, np.transpose(w)) #(x.shape[0],q)
        # 更新
        v = v + eta * np.dot(np.transpose(x), e) #(x.shape[1],q)
        w = w + eta * np.dot(np.transpose(b), g) #(q,1)
        gama = gama - eta * e.sum(axis=0) #(1,q)
        sigma = sigma - eta * g.sum(axis=0) #(1,1)
    return v, w, gama, sigma, losslist

def predict(x, v, w, gama, sigma):
    alpha = np.dot(x, v)
    b = sigmoid(alpha - gama)
    beta = np.dot(b, w)
    y_pred = np.round(sigmoid(beta - sigma))
    return y_pred

if __name__=='__main__':
    #设置参数
    q = 10
    eta=0.1
    max_iteration=500
    #读取数据
    data = pd.read_csv('xigua3.0.csv')
    data.drop('编号', axis=1, inplace=True)
    #数据预处理
    x,y = data_processing(data)
    '''
    留出法
    '''
    #留出法划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    #将数据转化为array，便于后续处理，返回
    x_train = np.array(x_train)
    y_train = np.array(y_train).reshape(y_train.shape[0], 1)
    x_test = np.array(x_test)
    y_test = np.array(y_test).reshape(y_test.shape[0], 1)
    # 标准bp
    v, w, gama, sigma, losslist_s= standard_bp(x_train, y_train, q, eta, max_iteration)
    # 误差可视化
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['kaiti']
    plt.plot([i + 1 for i in range(len(losslist_s))], losslist_s, label='标准BP', c='orange')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('标准BP网络误差变化')
    plt.legend()
    plt.savefig('loss_standard.png')
    plt.show()
    # 测试
    y_predict = predict(x_test, v, w, gama, sigma)
    print('留出法—标准BP算法的精度为：',np.mean(y_predict == y_test))
    precision = metrics.precision_score(y_test, y_predict)
    print("查准率为：", precision)
    recall = metrics.recall_score(y_test, y_predict)
    print("查全率为：", recall)
    f1 = 2 * precision * recall / (precision + recall)
    print("F1的值为：", f1)
    result = pd.DataFrame(np.hstack((y_test, y_predict)), columns=['真实值', '预测值'])
    result.to_csv('result_standard.csv', index=False)
    # 累积bp
    v, w, gama, sigma, losslist_c = cumulative_bp(x_train, y_train, q, eta, max_iteration)
    # 误差可视化
    plt.clf()
    plt.plot([i + 1 for i in range(len(losslist_c))], losslist_c, label='累积BP', c='orange')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('累积BP网络误差变化')
    plt.legend()
    plt.savefig('loss_cumulative.png')
    plt.show()
    # 测试
    y_predict = predict(x_test, v, w, gama, sigma)
    print('留出法-累积BP算法的精度为：',np.mean(y_predict == y_test))
    precision = metrics.precision_score(y_test, y_predict)
    print("查准率为：", precision)
    recall = metrics.recall_score(y_test, y_predict)
    print("查全率为：", recall)
    f1 = 2 * precision * recall / (precision + recall)
    print("F1的值为：", f1)
    result = pd.DataFrame(np.hstack((y_test, y_predict)), columns=['真实值', '预测值'])
    result.to_csv('result_cumulative.csv', index=False)
    #误差对比
    plt.clf()
    plt.plot([i + 1 for i in range(len(losslist_s))], losslist_s, label='标准BP', c='c')
    plt.plot([i + 1 for i in range(len(losslist_c))], losslist_c, label='累积BP', c='lightcoral')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('标准和累积BP网络误差变化')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

    '''
    留一法，测试其精度
    '''
    loo = LeaveOneOut()
    i = 0
    accuracy_s=0
    accuracy_c = 0
    for x_train_index, x_test_index in loo.split(x):
        x_train = x[x_train_index, :]
        y_train = y[x_train_index]
        x_test = x[x_test_index, :]
        y_test = y[x_test_index]
        x_train = np.array(x_train)
        y_train = np.array(y_train).reshape(y_train.shape[0], 1)
        x_test = np.array(x_test)
        y_test = np.array(y_test).reshape(y_test.shape[0], 1)
        # 用对率回归进行训练，拟合数据
        v_s, w_s, gama_s, sigma_s, losslist_s = standard_bp(x_train, y_train, q, eta, max_iteration)
        v_c, w_c, gama_c, sigma_c, losslist_c = cumulative_bp(x_train, y_train, q, eta, max_iteration)
        # 测试
        y_predict_s = predict(x_test, v_s, w_s, gama_s, sigma_s)
        y_predict_c = predict(x_test, v_c, w_c, gama_c, sigma_c)
        #累加精度
        accuracy_s += np.mean(y_predict_s == y_test)
        accuracy_c += np.mean(y_predict_c == y_test)
    # 计算精度
    accuracy_s = accuracy_s / x.shape[0]
    accuracy_c = accuracy_c / x.shape[0]
    print("留一法-标准BP算法的精度为：", accuracy_s)
    print("留一法-累积BP算法的精度为：", accuracy_c)





