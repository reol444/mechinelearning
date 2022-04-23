import numpy as np
import pandas as pd
import mypruning as mp
import plotTree
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
import queue
import myqueue


class Node:
    def __init__(self):
        self.property_name=None #节点属性名
        self.property_index=None #节点属性序号
        self.subtree={} #子树集合
        self.split_factor=None #划分因素的值，如信息增益的值
        self.is_continuous = False #属性是否连续
        self.split_value = None #属性划分值
        self.is_leaf = False #是否为叶节点
        self.leaf_class = None #叶节点的类别
        self.leaf_num = None #叶节点个数
        self.high = -1 #树的深度

class DecisionTree:
    #初始化
    def __init__(self,devide_method='gain',pruning_method=None):
        #判断输入的划分方法和剪枝方法是否合法
        assert devide_method in ('gain', 'gini','LR')
        assert pruning_method in (None, 'pre_pruning', 'post_pruning')
        #初始化
        self.devide_method = devide_method
        self.pruning_method = pruning_method
        self.columns = None  #包括原数据的列名
        self.tree = None #最终生成的树

    #生成树
    def build_tree(self, x, y, old_y,x_val, y_val, prop_dict, prop_continuous):
        #初始化当前节点
        cur_node=Node()
        #若均属于同一类别
        cur_node.leaf_num=0
        if y.nunique()==1:
            cur_node.is_leaf = True
            cur_node.leaf_class = y.values[0]
            cur_node.high = 0
            cur_node.leaf_num += 1
            return cur_node
        # 若属性已全部使用，或者样本属性取值相同，或划分为空集
        if x.empty:
            cur_node.is_leaf = True
            cur_node.leaf_class = pd.value_counts(old_y).index[0]
            cur_node.high = 0
            cur_node.leaf_num += 1
            return cur_node
        #选择最好的划分属性
        bestprop_name, best_split_factor = self.choose_bestprop(x, y, prop_continuous)
        print(bestprop_name, best_split_factor)
        #赋予当前节点信息
        cur_node.property_name = bestprop_name
        cur_node.split_factor = best_split_factor[0]
        cur_node.property_index = self.columns.index(bestprop_name)
        #当前属性的值
        prop_values = x.loc[:, bestprop_name]
        #验证集的值
        prop_values_val = x_val.loc[:, bestprop_name]
        if prop_continuous[bestprop_name] == 'not continuous':  # 若为离散值
            cur_node.is_continuous = False
            #找出当前属性的值
            unique_vals = prop_dict[bestprop_name]
            #从属性集中剔除当前属性
            sub_x = x.drop(bestprop_name, axis=1)
            #开始生成子节点
            max_high = -1
            if self.pruning_method =='pre_pruning':
                cur_node_pre=mp.pre_pruning_not_continuous(y, y_val, unique_vals, prop_values, prop_values_val, cur_node)
                if cur_node_pre:
                    return cur_node_pre
            for value in unique_vals: #对当前属性的每一个唯一值
                cur_node.subtree[value] = self.build_tree(sub_x[prop_values == value], y[prop_values == value],y,x_val[prop_values_val == value],y_val[prop_values_val == value],prop_dict,prop_continuous)  #递归生成子节点并放入子集中
                if cur_node.subtree[value].high > max_high:  # 记录子树下最高的高度
                    max_high = cur_node.subtree[value].high
                cur_node.leaf_num += cur_node.subtree[value].leaf_num #记录叶节点数量
            cur_node.high = max_high + 1
        else:  # 若为连续值
            cur_node.is_continuous = True
            cur_node.split_value = best_split_factor[1] #划分值
            greater_part = '>{:.2f}'.format(cur_node.split_value) #大于等于划分值的部分的名
            less_part = '<={:.2f}'.format(cur_node.split_value) #小于划分值的部分的名
            if self.pruning_method =='pre_pruning':
                cur_node_pre=mp.pre_pruning_continuous(y, y_val, prop_values, prop_values_val, cur_node)
                if cur_node_pre:
                    return cur_node_pre
            cur_node.subtree[greater_part] = self.build_tree(x[prop_values > cur_node.split_value],y[prop_values > cur_node.split_value],y,x_val[prop_values_val > cur_node.split_value],y_val[prop_values_val > cur_node.split_value],prop_dict,prop_continuous)#递归生成子节点并放入子集中
            cur_node.subtree[less_part] = self.build_tree(x[prop_values <= cur_node.split_value],y[prop_values <= cur_node.split_value],y,x_val[prop_values_val <= cur_node.split_value],y_val[prop_values_val <= cur_node.split_value],prop_dict,prop_continuous)#递归生成子节点并放入子集中
            cur_node.leaf_num += (cur_node.subtree[greater_part].leaf_num + cur_node.subtree[less_part].leaf_num) #记录叶节点个数
            cur_node.high = max(cur_node.subtree[greater_part].high, cur_node.subtree[less_part].high) + 1 #记录高度
        return cur_node

    def build_tree_bfs(self, x, y, prop_dict, prop_continuous):
        #初始化节点队列和数据队列
        nodequeue=queue.Queue()
        dataqueue=queue.Queue()
        #初始化根节点（树）
        tree=Node()
        #根节点和训练集入队
        nodequeue.put(tree)
        dataqueue.put([x, y])
        #记录每一层的非叶节点树，当前为1
        not_leaf=1
        #当节点队列不为空
        while not nodequeue.empty():
            sub_not_leaf=0 #下一层的非叶节点数
            for i in range(not_leaf): #对当前层的每一个非叶节点
                #节点与数据出队
                cur_node=nodequeue.get()
                cur_data=dataqueue.get()
                #得到x和y
                cur_x=cur_data[0]
                cur_y=cur_data[1]
                #找到最优属性
                bestprop_name, best_split_factor = self.choose_bestprop(cur_x, cur_y, prop_continuous)
                print(bestprop_name, best_split_factor)
                #赋予当前节点信息
                cur_node.property_name = bestprop_name
                cur_node.split_factor = best_split_factor[0]
                cur_node.property_index = self.columns.index(bestprop_name)
                #当前划分属性的属性值集和属性值
                prop_values = cur_x.loc[:, bestprop_name]
                unique_vals = prop_dict[bestprop_name]
                #对划分属性的每一个值
                for value in unique_vals:
                    print(value)
                    sub_node = Node()
                    cur_node.subtree[value]=sub_node #生成一个新节点作为子节点
                    #划分出子训练集 x和y
                    cur_x_value=cur_x[prop_values == value]
                    cur_y_value=cur_y[prop_values == value]
                    if len(cur_x_value)==0: #若Dv为空
                        sub_node.is_leaf = True #设置为叶节点
                        sub_node.leaf_class = cur_y.values[0] #类别为父结点中样本最多的类别
                        continue #跳过后续，继续循环
                    sub_cur_x=cur_x.drop(bestprop_name, axis=1)
                    if len(sub_cur_x)==0 or cur_y_value.nunique()==1: #若A/a*为空或者均为一类
                        sub_node.is_leaf = True #设置为叶节点
                        sub_node.leaf_class = cur_y_value.values[0] #类别为当前训练集中样本最多的类别
                        continue #跳过后续，继续循环
                    nodequeue.put(sub_node) #子节点入队
                    dataqueue.put([cur_x_value, cur_y_value]) #子训练集入队
                    sub_not_leaf += 1 #子层中非叶节点数加一
            not_leaf=sub_not_leaf #将下一层赋予当前层数
        return tree

    def build_tree_dfs(self, x, y, prop_dict, prop_continuous):
        # 初始化节点队列和数据队列
        nodequeue = myqueue.myqueue()
        dataqueue = myqueue.myqueue()
        propqueue = myqueue.myqueue()
        # 初始化根节点（树）
        tree = Node()
        # 根节点和训练集入队
        nodequeue.put(tree)
        dataqueue.put([x, y])
        # 当节点队列不为空
        while not nodequeue.empty():
            # 节点与数据出队
            cur_node = nodequeue.get()
            cur_data = dataqueue.get()
            # 得到x和y
            cur_x = cur_data[0]
            cur_y = cur_data[1]
            #若当前节点没有被划分过
            if cur_node.property_name==None:
                #进行划分
                bestprop_name, best_split_factor = self.choose_bestprop(cur_x, cur_y, prop_continuous)
                print(bestprop_name, best_split_factor)
                #赋予当前节点信息
                cur_node.property_name = bestprop_name
                cur_node.split_factor = best_split_factor[0]
                cur_node.property_index = self.columns.index(bestprop_name)
                cur_unique_vals = prop_dict[bestprop_name] #赋予当前划分值
            else: #否则从队列中中取出当前剩余的划分值
                cur_unique_vals = propqueue.get()
            # 当前划分属性的属性值集和当前选择的划分属性值
            prop_values = cur_x.loc[:, cur_node.property_name]
            value=cur_unique_vals[0]
            cur_unique_vals.pop(0) #将使用过的划分属性值剔除
            if not len(cur_unique_vals)==0: #若当前划分节点还有未使用的划分属性值，入队
                nodequeue.put(cur_node)
                dataqueue.put(cur_data)
                propqueue.put(cur_unique_vals)
            print(value)
            #建立子节点
            sub_node = Node()
            cur_node.subtree[value] = sub_node  # 生成一个新节点作为子节点
            # 划分出子训练集 x和y
            cur_x_value = cur_x[prop_values == value]
            cur_y_value = cur_y[prop_values == value]
            if len(cur_x_value) == 0:  # 若Dv为空
                sub_node.is_leaf = True  # 设置为叶节点
                sub_node.leaf_class = cur_y.values[0]  # 类别为父结点中样本最多的类别
                continue
            sub_cur_x = cur_x.drop(cur_node.property_name, axis=1)
            if len(sub_cur_x) == 0 or cur_y_value.nunique() == 1:  # 若A/a*为空或者均为一类
                sub_node.is_leaf = True  # 设置为叶节点
                sub_node.leaf_class = cur_y_value.values[0]  # 类别为当前训练集中样本最多的类别
                continue
            nodequeue.put(sub_node)  # 子节点入队
            dataqueue.put([cur_x_value, cur_y_value])  # 子训练集入队
        return tree

    #选择最好的划分节点
    def choose_bestprop(self, x, y,prop_continuous):
        #判断方法
        if self.devide_method == 'gini':
            return self.choose_bestprop_gini(x, y,prop_continuous)
        elif self.devide_method == 'gain':
            return self.choose_bestprop_gain(x, y, prop_continuous)
        elif self.devide_method == 'LR':
            return self.choose_bestprop_LR(x, y, prop_continuous)

    #信息熵
    def ent(self, y):
        p = pd.value_counts(y) / y.shape[0]  # 计算各类样本所占比率
        ent = np.sum(-p * np.log2(p)) #计算信息熵
        return ent

    #信息增益
    def gain(self, prop_values, y, entD, is_continuous=False):
        m = y.shape[0] #当前样本集合的样本个数
        unique_value = pd.unique(prop_values) #将属性值唯一化
        if not is_continuous: #若为离散值
            prop_ent = 0 #取值为av的样本集的信息熵
            for value in unique_value:
                Dv = y[prop_values == value]  # 当前特征中取值为value的样本，即书中的 Dv
                prop_ent += Dv.shape[0] / m * self.ent(Dv) #计算信息熵
            gain = entD - prop_ent  #计算信息增益
            return [gain]
        else:
            unique_value.sort()  # 将属性排序
            split_point_set = [(unique_value[i] + unique_value[i + 1]) / 2 for i in range(len(unique_value) - 1)] #计算Ta
            min_ent = float('inf')  # 最小信息熵
            min_point = None #最小信息熵点
            for split_point in split_point_set: #对每个分割点
                # 分为小于等于和大于两类
                Dv1 = y[prop_values <= split_point]
                Dv2 = y[prop_values > split_point]
                #计算每个分割点的信息熵
                prop_ent = Dv1.shape[0] / m * self.ent(Dv1) + Dv2.shape[0] / m * self.ent(Dv2)
                if prop_ent < min_ent: #找到信息熵最小点
                    min_ent = prop_ent
                    min_point = split_point
            gain = entD - min_ent #计算信息增益
            return [gain, min_point]


    #利用信息增益划分
    def choose_bestprop_gain(self, x, y,prop_continuous):
        prop = x.columns #获取属性集
        bestprop_name = None #最好的划分属性
        best_gain = [float('-inf')] #最好的信息增益
        entD = self.ent(y) #计算D的信息熵
        for prop_name in prop: #对每个属性
            is_continuous = prop_continuous[prop_name] == 'continuous' #利用sklearn库函数判断是否为连续值
            gain = self.gain(x[prop_name], y, entD, is_continuous) #调用信息增益计算函数
            if gain[0] > best_gain[0]: #找到最大的信息增益
                bestprop_name = prop_name #更新最佳划分属性
                best_gain = gain #更新增益

        return bestprop_name, best_gain

    # 计算基尼值
    def gini(self, y):
        p = pd.value_counts(y) / y.shape[0] #计算各类样本所占比率
        gini = 1 - np.sum(p ** 2) #计算基尼值
        return gini

    def gini_index(self, prop_values, y, is_continuous=False):
        m = y.shape[0] #当前样本集合的样本个数
        unique_value = pd.unique(prop_values)#将属性值唯一化
        if not is_continuous: #若为离散值
            gini_index = 0 #基尼指数
            for value in unique_value: #对每一个属性值
                Dv = y[prop_values == value] #当前属性中取值为value的样本，即书中的Dv
                m_dv = Dv.shape[0] #dv的样本个数
                gini = self.gini(Dv)  # 计算基尼指数
                gini_index += m_dv / m * gini
            return [gini_index]
        else:#若为连续值
            unique_value.sort()  #将属性排序
            split_point_set = [(unique_value[i] + unique_value[i + 1]) / 2 for i in range(len(unique_value) - 1)]#计算Ta
            min_gini = float('inf')#最小基尼指数
            min_gini_point = None #最好的分割点
            for split_point in split_point_set:  # 遍历所有的分割点，寻找基尼指数最小的分割点
                Dv1 = y[prop_values <= split_point] #小于等于当前分割点
                Dv2 = y[prop_values > split_point] #大于
                gini_index = Dv1.shape[0] / m * self.gini(Dv1) + Dv2.shape[0] / m * self.gini(Dv2) #计算基尼指数
                if gini_index < min_gini:#找到最小值
                    min_gini = gini_index #更新基尼指数
                    min_gini_point = split_point #更新划分点
            return [min_gini, min_gini_point]

    # 基尼指数划分
    def choose_bestprop_gini(self, x, y,prop_continuous):
        prop = x.columns #获取属性集
        bestprop_name = None #最好的划分属性
        best_gini = [float('inf')] #最好的基尼指数
        for prop_name in prop: #对每一个属性
            is_continuous = prop_continuous[prop_name] == 'continuous' #判断是否为连续
            gini_idex = self.gini_index(x[prop_name], y, is_continuous) #得到基尼指数
            if gini_idex[0] < best_gini[0]:#找到最小的基尼指数
                bestprop_name = prop_name #更新划分属性
                best_gini = gini_idex #更新最小的基尼指数
        return bestprop_name, best_gini

    def LR(self,prop_values, y,is_continuous):
        LR = LogisticRegression(solver='lbfgs',C=500,multi_class='ovr')
        if not is_continuous: #离散值
            x = pd.get_dummies(prop_values).values
            LR.fit(x, y)
            accuracy = LR.score(x, y)
            return [accuracy] #计算精度返回
        else: #连续值
            unique_value = pd.unique(prop_values)
            unique_value.sort()
            split_point_set = [(unique_value[i] + unique_value[i + 1]) / 2 for i in range(len(unique_value) - 1)]  # 计算Ta
            max_split_accuracy=float('-inf')
            best_split_point=None
            for split_point in split_point_set: #对每一个划分点
                x_split=prop_values>=split_point #划分
                x_split=pd.get_dummies(x_split).values
                LR.fit(x_split,y)
                accuracy = LR.score(x_split,y) #计算精度
                if accuracy>max_split_accuracy:
                    max_split_accuracy=accuracy
                    best_split_point=split_point #找到最佳划分点和精度
            return [max_split_accuracy, best_split_point] #返回


    def choose_bestprop_LR(self, x, y,prop_continuous):
        prop = x.columns  # 获取属性集
        bestprop_name = None  # 最好的划分属性
        best_acc = [float('-inf')]  #最高的精度
        for prop_name in prop:
            is_continuous = prop_continuous[prop_name] == 'continuous'  # 判断是否为连续
            accuracy=self.LR(x[prop_name], y,is_continuous)
            if accuracy[0]>best_acc[0]:
                bestprop_name = prop_name  # 更新划分属性
                best_acc = accuracy  # 更新最好的精度
        return bestprop_name, best_acc
    #学习
    def fit(self,x_train, y_train, x_val=None, y_val=None):
        #对训练集重新编号
        x_train.reset_index(inplace=True, drop=True)
        y_train.reset_index(inplace=True, drop=True)
        #对测试集重新编号
        if x_val is not None:
            x_val.reset_index(inplace=True, drop=True)
            y_val.reset_index(inplace=True, drop=True)
        self.columns = list(x_train.columns)  #训练集的属性名
        prop = x_train.columns  #获取属性集
        prop_dict={} #存储属性集的唯一值
        prop_continuous={} #存储属性是否连续
        for p in prop:
            prop_dict[p]=pd.unique(x_train.loc[:, p]).tolist() #计算得到唯一值
            l = len(prop_dict[p])
            if l>10: #若类别大于10个
                prop_continuous[p]='continuous' #认为连续
            else:
                prop_continuous[p] = 'not continuous' #否则不连续
        #构建树
        #递归实现
        self.tree = self.build_tree(x_train, y_train, y_train, x_val, y_val, prop_dict, prop_continuous)
        #广度优先
        # self.tree=self.build_tree_bfs(x_train, y_train,prop_dict, prop_continuous)
        #深度优先
        # self.tree = self.build_tree_dfs(x_train, y_train, prop_dict, prop_continuous)
        #后剪枝
        if self.pruning_method == 'post_pruning':
            mp.post_pruning(x_train, y_train, x_val, y_val, self.tree)
        return self

    def predict_single(self, x, subtree=None): #预测单维
        if subtree is None: #初始化
            subtree = self.tree
        if subtree.is_leaf: #判断是否到达叶节点
            return subtree.leaf_class #返回叶节点类
        if subtree.is_continuous:   #判断是否为连续值
            #是则进行连续值预测
            if x[subtree.property_index] >= subtree.split_value: #分支
                return self.predict_single(x, subtree.subtree['>{:.2f}'.format(subtree.split_value)]) #递归调用
            else:
                return self.predict_single(x, subtree.subtree['<={:.2f}'.format(subtree.split_value)]) #递归调用
        else: #为离散值
            return self.predict_single(x, subtree.subtree[x[subtree.property_index]]) #递归调用

    def predict(self, x): #进行预测
        if x.ndim == 1: #若x维数为1，直接调用predict_single
            return self.predict_single(x)
        else:
            return x.apply(self.predict_single, axis=1) #否则对每一维调用predict_single

if __name__ == '__main__':
    split_method='gain'
    # pruning='pre_pruning'
    pruning = 'post_pruning'
    # pruning = None
    #习题2和3
    # #西瓜2.0
    # data = pd.read_csv('xigua2.0.txt', index_col=0)
    # train = [1, 2, 3, 6, 7, 10, 14, 15, 16, 17]
    # train = [i - 1 for i in train]
    # x_train = data.iloc[train, :6]
    # y_train = data.iloc[train, 6]
    # test = [4, 5, 8, 9, 11, 12, 13]
    # test = [i - 1 for i in test]
    # x_val = data.iloc[test, :6]
    # y_val = data.iloc[test, 6]
    # tree = DecisionTree(split_method,pruning)
    # tree.fit(x_train, y_train, x_val, y_val)
    # print('精度为：',np.mean(tree.predict(x_val) == y_val))
    # # plotTree.create_plot(tree.tree)

    # #习题1
    # #iris
    # data = pd.read_csv('iris.csv', names=['属性1','属性2','属性3','属性4','类别'], header=None)
    # x = data.iloc[:, :4]
    # y = data.iloc[:, 4]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=0)
    # tree = DecisionTree(split_method, pruning)
    # tree.fit(x_train, y_train, x_val, y_val)
    # y_pred = tree.predict(x_test)
    # print("留出法的精度为：", np.mean(y_pred == y_test))
    # precision = metrics.precision_score(y_test, y_pred, average='macro')
    # print("查准率为：", precision)
    # recall = metrics.recall_score(y_test, y_pred, average='macro')
    # print("查全率为：", recall)
    # f1 = 2 * precision * recall / (precision + recall)
    # print("macro-F1的值为：", f1)
    # plotTree.create_plot(tree.tree)

    #wine
    data = pd.read_csv('wine.csv', names=['属性1', '属性2', '属性3', '属性4','属性5', '属性6', '属性7', '属性8','属性9', '属性10', '属性11','属性12','属性13','类别'], header=None)
    x = data.iloc[:, :13]
    y = data.iloc[:, 13]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, test_size=0.3, random_state=0)
    tree = DecisionTree(split_method,pruning)
    tree.fit(x_train, y_train, x_val, y_val)
    y_pred=tree.predict(x_test)
    print("留出法的精度为：",np.mean(y_pred == y_test))
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    print("查准率为：", precision)
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    print("查全率为：", recall)
    f1 = 2 * precision * recall / (precision + recall)
    print("macro-F1的值为：", f1)
    plotTree.create_plot(tree.tree)


    # # balloons
    # data = pd.read_csv('adult+stretch.csv',names=['属性1', '属性2', '属性3', '属性4','类别'], header=None)
    # x = data.iloc[:, :4]
    # y = data.iloc[:, 4]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=0)
    # tree = DecisionTree(split_method, pruning)
    # tree.fit(x_train, y_train, x_val, y_val)
    # y_pred = tree.predict(x_test)
    # print(y_test,y_pred)
    # print("留出法的精度为：", np.mean(y_pred == y_test))
    # precision = metrics.precision_score(y_test, y_pred, pos_label='T')
    # print("查准率为：", precision)
    # recall = metrics.recall_score(y_test, y_pred, pos_label='T')
    # print("查全率为：", recall)
    # f1 = 2 * precision * recall / (precision + recall)
    # print("F1的值为：", f1)
    # plotTree.create_plot(tree.tree)

    # # COVID-19 Surveillance Data Set
    # data = pd.read_csv('Surveillance.csv')
    # x = data.iloc[:, :7]
    # y = data.iloc[:, 7]
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=0)
    # tree = DecisionTree(split_method, pruning)
    # tree.fit(x_train, y_train, x_val, y_val)
    # y_pred = tree.predict(x_test)
    # print(y_pred)
    # print(y_test)
    # print("留出法的精度为：", np.mean(y_pred == y_test))
    # precision = metrics.precision_score(y_test, y_pred, average='macro')
    # print("查准率为：", precision)
    # recall = metrics.recall_score(y_test, y_pred, average='macro')
    # print("查全率为：", recall)
    # f1 = 2 * precision * recall / (precision + recall)
    # print("macro-F1的值为：", f1)
    # plotTree.create_plot(tree.tree)

    kf = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)
    accuracy=0
    for x_train_index, x_test_index in kf.split(x):
        x_train = x.iloc[x_train_index,:]
        y_train = y.iloc[x_train_index,]
        x_test = x.iloc[x_test_index,:]
        y_test = y.iloc[x_test_index,]
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
        tree = DecisionTree(split_method,pruning)
        tree.fit(x, y, x_val, y_val)
        y_pred=tree.predict(x_test)
        print(np.mean(y_pred == y_test))
        accuracy+=np.mean(y_pred == y_test)
    print('10折交叉验证法的精度为：',accuracy/10)
