import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

class SVM:
    #构造函数
    def __init__(self,max_iter,kernel,tol=0.001):
        self.max_iter=max_iter
        self.kernel_name=kernel
        self.tol=tol

    #初始化参数
    def init_model(self,x,y):
        self.x=x #属性集
        self.y=y #真实值
        self.b=0.0 #超平面截距b
        self.m=x.shape[0] #训练集的大小
        self.n=x.shape[1] #属性的个数
        self.alpha=np.zeros(self.m) #拉格朗日乘子
        self.E=[self.calculate_E(i) for i in range(self.m)] #smo中所需的E的集合，即预测值与真实值之间的差的集合
        self.C=10000.0 #松弛变量
        self.support_vectors=None #支持向量
        self.satisfy_index_list=None #满足0<a<C的样本

    #核函数的定义
    def kernel(self, x1, x2):
        if self.kernel_name == 'linear': #线性核，为点积
            return np.dot(x1, x2)
        elif self.kernel_name == 'Gaussian': #高斯核，设gama为0.5
            gama = 0.5
            return np.exp(-gama * np.linalg.norm(x2 - x1, ord=2)**2)
        elif self.kernel_name == 'Polynomial':  #多项式核，设次数为3次
            return np.power(np.dot(x2, x1.T), 3)
        return 0

    #预测值
    def f(self,x):
        fx=0
        for i in range(self.m):
            fx+=self.alpha[i] * self.y[i] * self.kernel(self.x[i],x)
        fx+=self.b
        return fx

    #计算Ei，为预测值与真实值之间的差
    def calculate_E(self,i):
        return self.f(self.x[i]) - self.y[i]

    #KKT条件，判断第i个样例是否符合条件
    def KKT(self, i):
        y_E=self.y[i]*self.E[i]
        if self.alpha[i] == 0:
            return y_E >= -self.tol
        elif 0 < self.alpha[i] < self.C:
            return abs(y_E) <= self.tol
        else:
            return y_E <= self.tol

    #对计算出的a2进行修剪
    def clip(self, alpha, L, H):
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha

    def fit(self,x,y):
        """
        开始训练，使用SMO算法进行迭代求解
        主要思想为：
        首先，寻找两个a,先找违背KKT条件最大的a,再寻找使到使|E1-E2|最大的a
        具体做法为:首先遍历所有满足0<alpha<C的样例,看其是否满足KKT条件,若是则选择第二个a,即找使到使|E1-E2|最大的a
        否则,遍历所有样本点,看其是否满足KKT条件.
        当两个变量均选择完毕，则对两个变量a以及b和Ei进行更新
        直到达到迭代次数或者所有样本点均满足KTT条件为止
        """
        self.init_model(x,y)
        flag = True #判断是否进行遍历，由于我们初始化a和b均为0，所以第一次必定遍历全部样本
        #外循环，开始遍历
        for ite in range(self.max_iter):
            #更新所有0<a<C的样本下标列表
            self.satisfy_index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
            success_update = 0
            if flag: #若遍历全部样本
                for i in range(self.m):
                    if self.inner_loop(i):
                        success_update += 1
            else: #若遍历满足0<a<C的样本
                for i in self.satisfy_index_list:
                    if self.inner_loop(i):
                        success_update += 1
            if flag: #判断当前遍历类型
                if success_update == 0:  #若遍历所有样本均未更新成功，停止迭代
                    print('由于所有样本均满足KTT条件，迭代提前终止，此时迭代次数为：',ite)
                    break
                flag = False
            elif success_update == 0: #若遍历所有满足0<a<C的样本，均未更新成功，则遍历所有样本
                flag = True
        self.support_vectors = [i for i in range(self.m) if self.alpha[i] > 0] #得到支持向量
    #内循环
    def inner_loop(self,i):
        #判断是否满足KKT条件，不满足则进行下述操作
        if not self.KKT(i):
            j = None
            #首先，从满足0<a<C的样本中寻找一个使|E1-E2|最大的点作为j,尝试更新
            if not len(self.satisfy_index_list) == 0:
                j=self.select_j(i)
                if self.update_alpha(i, j):
                    return True
            #若是上述更新失败，则遍历所有满足0<a<C的样本，尝试更新
            for k in np.random.permutation(np.array(self.satisfy_index_list)):
                if k == j:
                    continue
                if self.update_alpha(i, k):
                    return True
            #若再次更新失败，则遍历所有样本，尝试更新
            for k in np.random.permutation(self.m):
                if k in self.satisfy_index_list:
                    continue
                if self.update_alpha(i, k):
                    return True
        #若全部更新失败，则放弃当前a1，返回False
        return False

    #选择j
    def select_j(self,i):
        maxdE=float('-inf')
        for j in self.satisfy_index_list:
            if abs(self.E[i]-self.E[j])>maxdE:
                maxdE=abs(self.E[i]-self.E[j])
                best_j=j
        return best_j

    #更新a
    def update_alpha(self,i,j):
        if i==j:
            return False
        #更新a2的边界
        if self.y[i] == self.y[j]:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        if L == H:
            return False
        #得到E
        E1 = self.E[i]
        E2 = self.E[j]
        #求出miu
        eta = self.kernel(self.x[i], self.x[i]) + self.kernel(self.x[j], self.x[j]) - 2 * self.kernel(self.x[i],self.x[j])
        #若小于等于0,有对应的计算方法,但过于复杂,这里直接当作更新失败
        if eta <= 0:
            return False
        #计算新的a2并修剪
        alpha_2_new = self.alpha[j] + self.y[j] * (E1 - E2) / eta
        alpha_2_new = self.clip(alpha_2_new, L, H)
        #a2没有足够的改变，更新失败。
        if abs(alpha_2_new - self.alpha[j]) < 0.00001:
            return False
        # 计算a1
        alpha_1_new = self.alpha[i] + self.y[i] * self.y[j] * (self.alpha[j] - alpha_2_new)
        # 更新a1, a2
        self.alpha[i] = alpha_1_new
        self.alpha[j] = alpha_2_new
        # 更新b
        b1_new = -E1 - self.y[i] * self.kernel(self.x[i], self.x[i]) * (alpha_1_new - self.alpha[i]) \
                 - self.y[j] * self.kernel(self.x[j], self.x[i]) * (alpha_2_new - self.alpha[j]) + self.b
        b2_new = -E2 - self.y[i] * self.kernel(self.x[i], self.x[j]) * (alpha_1_new - self.alpha[i]) \
                 - self.y[j] * self.kernel(self.x[j], self.x[j]) * (alpha_2_new - self.alpha[j]) + self.b
        if 0 < alpha_1_new < self.C:
            self.b = b1_new
        elif 0 < alpha_2_new < self.C:
            self.b = b2_new
        else:
            self.b = (b1_new + b2_new) / 2
        #更新E
        self.update_E()
        return True

    #更新E，当a为0时，没有影响，故更新可以写成如下形式
    def update_E(self):
        for i in range(self.m):
            self.E[i]=self.calculate_E(i)

    #预测
    def predict(self, x):
        y_pred=[]
        for xi in x:
            fxi=self.f(xi)
            if fxi > 0:
                y_pred.append(1)
            else:
                y_pred.append(-1)
        return np.array(y_pred)

    #画图
    def plot(self):
        ax = plt.subplot()
        y = self.y
        #正例反例下标列表
        good = []
        bad = []
        #遍历加入到对应类别的列表中
        for i in range(len(y)):
            if y[i] == 1:
                good.append(i)
            else:
                bad.append(i)
        #为密度和含糖率设置坐标轴和长度
        x_axes1 = np.linspace(0, 0.8, 600)
        x_axes2 = np.linspace(0, 0.6, 600)
        #生成矩阵格
        x1,x2 = np.meshgrid(x_axes1, x_axes2)
        #预测y
        y_pred = self.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)
        #绘制分类决策线，利用contour等高线来解决，[0]表示等高线高度为0，即我们的决策线
        CS = ax.contour(x1, x2, y_pred, [0], colors='orange', linewidths=1)
        #画出正例反例
        ax.scatter(self.x[good, 0], self.x[good, 1], label='good', color='g')
        ax.scatter(self.x[bad, 0], self.x[bad, 1], label='bad', color='r')
        #圈出支持向量
        ax.scatter(x[self.support_vectors, 0], x[self.support_vectors, 1], marker='o', c='None', edgecolors='gray', s=150,
                   label='support_vectors')
        #标题
        ax.set_title('watermelon3.0a-{} kernel'.format(self.kernel_name))
        ax.legend()
        plt.savefig('watermelon3.0a-{}_kernel.png'.format(self.kernel_name))
        plt.show()



if __name__ == '__main__':
    #读取西瓜数据集3.0a
    data = pd.read_csv('watermelon3.0a.csv')
    x = data.iloc[:, [1, 2]].values
    y = data.iloc[:, 3].values
    #将坏瓜标签处理为-1
    y[y == 0] = -1
    #初始化支持向量机，设置最大迭代次数为1000，核函数随意
    svm = SVM(max_iter=1000, kernel='Gaussian')
    #训练
    svm.fit(x,y)
    #预测
    y_pred=svm.predict(x)
    #输出支持向量机的信息
    print('alpha的值为：')
    print(svm.alpha)
    print('b的值为：',svm.b)
    print('训练集精度为：',np.mean(y_pred==y))
    print('支持向量为：',svm.support_vectors)
    #画图
    svm.plot()