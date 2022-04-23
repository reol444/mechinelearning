import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#读入西瓜数据
wm_data=pd.read_excel('watermelon3.0a.xlsx')
#获得x
x=np.array([list(wm_data['密度']),list(wm_data['含糖率'])]).T
#根据好瓜坏瓜分类，前八个是好瓜，后面是坏瓜
x1 = np.array(x[:8])
x2 = np.array(x[8:])
#求正反例均值
u1 = np.mean(x1, axis=0)
u2 = np.mean(x2, axis=0)
#求出w
sw = np.dot((x1-u1).T,(x1-u1)) + np.dot((x2-u2).T,(x2-u2))
w = np.mat(sw).I * (u1.reshape((-1, 1)) - u2.reshape((-1, 1)))
print(w)
#画出原始点
plt.scatter(x1[:, 0], x1[:, 1], c='green', label='good')
plt.scatter(x2[:, 0], x2[:, 1], c='red', label='bad')
#分别将正例和反例的投影点画出
w = w / np.linalg.norm(w)
x1_p=np.dot(np.dot(x1,w),w.T)
x2_p=np.dot(np.dot(x2,w),w.T)
plt.scatter(x1_p[:,0].tolist(),x1_p[:,1].tolist(),c='c',label='good_proj')
plt.scatter(x2_p[:,0].tolist(),x2_p[:,1].tolist(),c='m',label='bad_proj')
#画出投影线，以便观察
for i in range(x1.shape[0]):
    plt.plot([x1[i, 0], x1_p[i, 0]], [x1[i, 1], x1_p[i, 1]], '--b', linewidth=0.2)
for i in range(x2.shape[0]):
    plt.plot([x2[i, 0], x2_p[i, 0]], [x2[i, 1], x2_p[i, 1]], '--b', linewidth=0.2)
#画出LDA直线
x1 = np.linspace(0,0.2,100)
k=w[1]/w[0]
x2 = k[0,0]*x1
plt.plot(x1,x2,label='w')
#设置标签和大小
plt.axis("equal")
plt.xlabel('density')
plt.ylabel('suger')
plt.title('LDA-watermelen3.0a')
plt.legend()
plt.show()
plt.clf()
u1_p=np.dot(np.dot(u1,w),w.T).tolist()[0]
u2_p=np.dot(np.dot(u2,w),w.T).tolist()[0]
print("正例在直线上的投影点为：",u1_p)
print("反例在直线上的投影点为：",u2_p)
plt.scatter(u1[0],u1[1],c='green',label='mean_good')
plt.scatter(u2[0],u2[1],c='red',label='mean_bad')
plt.scatter(u1_p[0],u1_p[1],c='c',label='mean_good_proj')
plt.scatter(u2_p[0],u2_p[1],c='m',label='mean_bad_proj')
#画出投影线，以便观察
plt.plot([u1[0], u1_p[0]], [u1[1], u1_p[1]], '--b', linewidth=0.2)
plt.plot([u2[0], u2_p[0]], [u2[1], u2_p[1]], '--b', linewidth=0.2)
x1 = np.linspace(0,0.2,100)
k=w[1]/w[0]
x2 = k[0,0]*x1
plt.plot(x1,x2,label='w')
#设置标签和大小
plt.axis("equal")
plt.xlabel('density')
plt.ylabel('suger')
plt.title('LDA-watermelen3.0a_mean')
plt.legend()
plt.show()