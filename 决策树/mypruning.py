import pandas as pd
import numpy as np

def set_leaf(leaf_class, tree):
    # 设置节点为叶节点
    tree.is_leaf = True
    tree.leaf_class = leaf_class
    tree.property_name = None  # 节点属性名
    tree.property_index = None  # 节点属性序号
    tree.subtree = {}
    tree.split_factor = None
    tree.split_value = None
    tree.high = 0
    tree.leaf_num = 1

def post_pruning(x_train, y_train, x_val, y_val, tree=None):
    if tree.is_leaf: return tree #根节点为叶节点，直接返回
    if x_val.empty:  return tree #验证集为空集时，返回
    most_train=pd.value_counts(y_train).index[0] #当前样本最多的类别
    accuracy = np.mean(y_val == most_train)  #当前节点下验证集样本准确率
    if not tree.is_continuous:
        high = -1 #高度
        tree.leaf_num = 0 #叶节点个数
        is_sub_leaf = True  #判断当前所有子树是否都为叶节点，默认为真
        for prop in tree.subtree.keys():
            #对每一个子节点
            train = x_train.loc[:, tree.property_name] == prop #划分出当前节点的训练集
            val = x_val.loc[:, tree.property_name] == prop #划分出当前节点的验证集
            #递归进行剪枝
            tree.subtree[prop]=post_pruning(x_train[train], y_train[train], x_val[val], y_val[val], tree.subtree[prop])
           #找到子结点中最深的深度
            if tree.subtree[prop].high > high:
                high = tree.subtree[prop].high
            #更新当前节点的叶节点数量
            tree.leaf_num += tree.subtree[prop].leaf_num
            #判断当前节点的子节点是否都为叶节点
            if not tree.subtree[prop].is_leaf:
                is_sub_leaf = False
        #更新当前节点的高度
        tree.high = high + 1
        #若子节点全部为叶节点
        if is_sub_leaf:
            #对验证集的结果进行判断，得到划分后的结果
            correct_val = y_val.groupby(x_val.loc[:, tree.property_name]).apply(lambda x: np.sum(x == tree.subtree[x.name].leaf_class))
            #得到划分后的正确率
            split_accuracy = correct_val.sum() / y_val.shape[0]
            print(accuracy, split_accuracy)
            #与划分前的正确率进行比较
            if accuracy >= split_accuracy:  #若划分前的正确率大于划分后的，进行剪枝操作，将当前节点设置为叶节点
                set_leaf(pd.value_counts(y_train).index[0], tree)
    else:
        greater_train = x_train.loc[:, tree.property_name] > tree.split_value; #训练集中大于等于划分值的部分
        less_train = x_train.loc[:, tree.property_name] <= tree.split_value; #训练集中小于划分值的部分
        greater_val = x_val.loc[:, tree.property_name] > tree.split_value #验证集中大于等于划分值的部分
        less_val = x_val.loc[:, tree.property_name] <= tree.split_value #验证集中小于于划分值的部分
        #对大于等于划分值的子树进行后剪枝，递归
        greater_subtree = post_pruning(x_train[greater_train], y_train[greater_train], x_val[greater_val],y_val[greater_val],tree.subtree['>{:.2f}'.format(tree.split_value)])
        #更新子树
        tree.subtree['>{:.2f}'.format(tree.split_value)] = greater_subtree
        # 对小于划分值的子树进行后剪枝，递归
        less_subtree = post_pruning(x_train[less_train], y_train[less_train],x_val[less_val], y_val[less_val],tree.subtree['<={:.2f}'.format(tree.split_value)])
        # 更新子树
        tree.subtree['<={:.2f}'.format(tree.split_value)] = less_subtree
        #更新树高和叶节点数量
        tree.high = max(greater_subtree.high, less_subtree.high) + 1
        tree.leaf_num = (greater_subtree.leaf_num + less_subtree.leaf_num)
        #若两个子节点均为为叶节点
        if greater_subtree.is_leaf and less_subtree.is_leaf:
            #判断当前属性值的划分函数
            def split_fun(x):
                if x >= tree.split_value:
                    return '>{:.2f}'.format(tree.split_value) #大于等于
                else:
                    return '<={:.2f}'.format(tree.split_value) #小于
            # 对验证集的结果进行判断，得到划分后的结果
            correct_val = y_val.groupby(x_val.loc[:, tree.property_name].map(split_fun)).apply(lambda x: np.sum(x == tree.subtree[x.name].leaf_class))
            #得到划分后的正确率
            split_accuracy = correct_val.sum() / y_val.shape[0]
            print(accuracy, split_accuracy)
            # 与划分前的正确率进行比较
            if accuracy >= split_accuracy:  #若划分前的正确率大于划分后的，进行剪枝操作，将当前节点设置为叶节点
                set_leaf(pd.value_counts(y_train).index[0], tree)
    return tree

def pre_pruning_not_continuous(y_train, y_val, unique_vals, prop_values, prop_values_val, cur_node):
    if y_val.empty: return False
    # 当前节点划分前验证集样本的正确率
    most = pd.value_counts(y_train).index[0]
    accuracy = np.mean(y_val == most)
    #当前验证集划分后的正确个数
    correct_sum = 0
    for value in unique_vals: #对划分后的每一个属性
        y_tarin_split = y_train[prop_values == value]
        if len(y_tarin_split) == 0:
            most_val_split = pd.value_counts(y_train).index[0] #若所包含的训练集为空，设置为父结点中训练集最多类别
        else:
            most_val_split = pd.value_counts(y_tarin_split).index[0] #否则，设置为当前结点训练集的最多类别
        split_result = y_val[prop_values_val == value] == most_val_split #计算验证集分类结果
        if len(split_result) == 0: #可能验证集分类后为空，继续
            continue
        for correct in split_result: #计算正确的个数，累加
            if correct:
                correct_sum += 1
    split_accuracy = correct_sum / y_val.shape[0] #计算划分后的正确率
    print(accuracy, split_accuracy)
    if accuracy >= split_accuracy: #若划分前大于等于划分后，将当前节点设置为叶节点，返回当前节点
        set_leaf(pd.value_counts(y_train).index[0], cur_node)
        return cur_node
    else:
        return False

def pre_pruning_continuous(y_train, y_val, prop_values, prop_values_val, cur_node):
    if y_val.empty: return False
    # 当前节点划分前验证集样本的正确率
    most = pd.value_counts(y_train).index[0]
    accuracy = np.mean(y_val == most)
    # 当前验证集划分后的正确个数
    correct_sum = 0
    y_train_greater=y_train[prop_values > cur_node.split_value] #划分出D+
    if len(y_train_greater) == 0:
        most_val_greater = pd.value_counts(y_train).index[0] #若所包含的训练集为空，设置为父结点中训练集最多类别
    else:
        most_val_greater = pd.value_counts(y_train_greater).index[0] #否则，设置为当前结点训练集的最多类别
    split_result_greater = y_val[prop_values_val > cur_node.split_value] == most_val_greater #计算验证集分类结果
    if not len(split_result_greater) == 0: #计算正确的个数，累加
        for correct in split_result_greater:
            if correct:
                correct_sum += 1
    y_train_less = y_train[prop_values <= cur_node.split_value] #划分出D-
    if len(y_train_less) == 0:
        most_val_less = pd.value_counts(y_train).index[0] #若所包含的训练集为空，设置为父结点中训练集最多类别
    else:
        most_val_less = pd.value_counts(y_train_less).index[0] #否则，设置为当前结点训练集的最多类别
    split_result_less = y_val[prop_values_val <= cur_node.split_value] == most_val_less #计算验证集分类结果
    if not len(split_result_less) == 0: #计算正确的个数，累加
        for correct in split_result_less:
            if correct:
                correct_sum += 1
    split_accuracy = correct_sum / y_val.shape[0] #计算划分后的正确率
    print(accuracy, split_accuracy)
    if accuracy >= split_accuracy: #若划分前大于等于划分后，将当前节点设置为叶节点，返回当前节点
        set_leaf(pd.value_counts(y_train).index[0], cur_node)
        return cur_node
    else:
        return False