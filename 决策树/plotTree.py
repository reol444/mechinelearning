from matplotlib import pyplot as plt
import matplotlib

decision_node = dict(boxstyle='round,pad=0.3', fc='w') #设置框的格式，圆角矩形，为属性框
leaf_node = dict(boxstyle='circle,pad=0.3', fc='w') #设置框的格式，圆，为叶节点
arrow_args = dict(arrowstyle="<-") #箭头格式

y_off = None
x_off = None
total_num_leaf = None
total_high = None

def plot_node(node_text, center_pt, parent_pt, node_type, ax):
    ax.annotate(node_text, # 注释文本的内容
                xy=[parent_pt[0], parent_pt[1] - 0.02], #xy 被注释的坐标点，二维元组形如(x,y)
                xycoords='axes fraction', #xycoords 被注释点的坐标系属性，以子绘图区左下角为参考，单位是百分比
                xytext=center_pt, #xytext 注释文本的坐标点
                textcoords='axes fraction', #textcoords注释文本的坐标系属性，以子绘图区左下角为参考，单位是百分比
                va="center",
                ha="center",
                size=15,
                bbox=node_type, #bbox，在文本周围绘制一个框
                arrowprops=arrow_args #arrowprops箭头，注释文本和被注释点之间画一个箭头
                )

def plot_mid_text(mid_text, center_pt, parent_pt, ax):
    x_mid = (parent_pt[0] - center_pt[0]) / 2 + center_pt[0]
    y_mid = (parent_pt[1] - center_pt[1]) / 2 + center_pt[1]
    ax.text(x_mid, y_mid, mid_text, fontdict=dict(size=10))


def plot_tree(my_tree, parent_pt, node_text, ax_):
    global y_off
    global x_off
    global total_num_leaf
    global total_high

    num_of_leaf = my_tree.leaf_num
    center_pt = (x_off + (1 + num_of_leaf) / (2 * total_num_leaf), y_off)

    plot_mid_text(node_text, center_pt, parent_pt, ax_)

    if total_high == 0:  # total_high为零时，表示就直接为一个叶节点。因为西瓜数据集的原因，在预剪枝的时候，有时候会遇到这种情况。
        plot_node(my_tree.leaf_class, center_pt, parent_pt, leaf_node, ax_)
        return
    plot_node(my_tree.property_name, center_pt, parent_pt, decision_node, ax_)

    y_off -= 1 / total_high
    for key in my_tree.subtree.keys():
        if my_tree.subtree[key].is_leaf:
            x_off += 1 / total_num_leaf
            plot_node(str(my_tree.subtree[key].leaf_class), (x_off, y_off), center_pt, leaf_node, ax_)
            plot_mid_text(str(key), (x_off, y_off), center_pt, ax_)
        else:
            plot_tree(my_tree.subtree[key], center_pt, str(key), ax_)
    y_off += 1 / total_high


def create_plot(tree):
    global y_off
    global x_off
    global total_num_leaf
    global total_high

    total_num_leaf = tree.leaf_num
    total_high = tree.high
    y_off = 1
    x_off = -0.5 / total_num_leaf

    fig, a = plt.subplots()
    a.set_xticks([])  # 隐藏坐标轴刻度
    a.set_yticks([])
    a.spines['right'].set_color('none')  # 设置隐藏坐标轴
    a.spines['top'].set_color('none')
    a.spines['bottom'].set_color('none')
    a.spines['left'].set_color('none')
    plot_tree(tree, (0.5, 1), '', a)
    matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
    plt.show()
