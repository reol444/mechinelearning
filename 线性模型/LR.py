import numpy as np

def LR(x,y,beta,N):
    #迭代准备
    beta_list=[]
    n=0 #迭代次数，初值为0
    while n<N:
        n += 1
        beta_list.append(beta.T[0].tolist())
        # 计算betaT*x，即wT*x+b
        betaT_x = np.array(np.dot(beta.T[0],x))
        #一阶导数，初值为0
        dbeta=0
        #二阶导数，初值为0
        d2beta=0
        #循环计算一二阶导
        for i in range(len(y)):
            #求导准备
            xi=np.array([x[:,i]]).T
            xiT=np.array([x[:,i]])
            p1 = np.exp(betaT_x[i])/(1 + np.exp(betaT_x[i]))
            #求导
            dbeta-=np.dot(xi, (y[i] - p1)) #P60的3.30式，一阶导数
            d2beta+=np.dot(xi,xiT) * p1 * (1 - p1) #P60的3.31式，二阶导数
        #P59的3.29式，牛顿迭代的更新公式
        beta=beta-np.dot(np.linalg.inv(d2beta),dbeta)
    return beta

