#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: KALI LINUX
@编译器：python3.7
@Created on 2018-11-08 22:33
@Author:ChileWang
@algorithm：决策树可视化
"""
import matplotlib.pyplot as plt


decision_node = dict(boxstyle='sawtooth', fc='0.8')  # 定义节点文本框
leaf_node = dict(boxstyle='round4', fc='0.8')  # 定义叶子节点文本框
arrow_args = dict(arrowstyle='<-')  # 定义箭头格式


def plot_node(node_txt, center_pt, parent_pt, node_type):  # 绘制箭头注释
    """

    | 参数 | 坐标系 |
| 'figure points' | 距离图形左下角的点数量 |
| 'figure pixels' | 距离图形左下角的像素数量 |
| 'figure fraction' | 0,0 是图形左下角，1,1 是右上角 |
| 'axes points' | 距离轴域左下角的点数量 |
| 'axes pixels' | 距离轴域左下角的像素数量 |
| 'axes fraction' | 0,0 是轴域左下角，1,1 是右上角 |
| 'data' | 使用轴域数据坐标系 |

    xy=(横坐标，纵坐标)  箭头尖端
    xytext=(横坐标，纵坐标) 文字的坐标，指的是最左边的坐标
    arrowprops= {
        facecolor= '颜色',
        shrink = '数字' <1  收缩箭头
    }

    node_txt: 节点的名字；
    xy: 被注解的东西的位置，在决策树中为上一个节点的位置
    xycoords:被注解的东西依据的坐标原点位置，是以图像还是坐标轴
    xytext: 注解内容的中心坐标 # textcoords:注解内容依据的坐标原点位置
    ha: horizontal alignment 文本中内容竖向对齐方式
    va: vertical alignment 文本中的内容 横向对齐方式
    arrowprops: 标记线的类型，是一个字典，如果字典中包含key为arrowstyle的，则默认类别有'->'等 # bbox: 对方框的设置

    :param node_txt:节点文本
    :param center_pt:注释内容的坐标
    :param parent_pt:被注释的内容的坐标
    :param node_type:节点类型
    :return:
    """
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                            xytext=center_pt, textcoords='axes fraction',
                            va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('a leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


def get_leafs_num(my_tree):  # 获得叶子数目
    """
    :param my_tree:
    :return: 叶子数目
    """
    leaf_nums = 0
    first_str = ''
    for key in my_tree.keys():
        first_str = key
        break

    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            leaf_nums += get_leafs_num(second_dict[key])
        else:
            leaf_nums += 1

    return leaf_nums


def get_tree_depth(my_tree):
    """
    :param my_tree:
    :return: 树的深度
    """
    max_depth = 0
    first_str = ''
    for key in my_tree.keys():
        first_str = key
        break
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrieve_tree(i):
    """
    :param i:
    :return: 返回选中的树
    """
    tree_list = [
        {
            'no surfacing': {0: 'no', 1: {'flipper': {0: 'no', 1: 'yes'}}}
        },
        {
            'no surfacing': {0: 'no', 1: {'flipper': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}
        }
    ]
    return tree_list[i]


def plot_mid_text(cntr_pt, parent_pt, txt_str):
    """
    在父子节点中填充文本信息
    :param cntr_pt: 子节点
    :param parent_pt: 父节点
    :param txt_str: 文本信息
    :return:
    """
    x_mid = (float(parent_pt[0]) + float(cntr_pt[0])) / 2.0
    y_mid = (float(parent_pt[1]) + float(cntr_pt[1])) / 2.0
    create_plot.ax1.text(x_mid, y_mid, txt_str)


def plot_tree(my_tree, parent_pt, node_txt):
    leaf_nums = get_leafs_num(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = ''
    for key in my_tree.keys():
        first_str = key
        break
    cntrpt = (plot_tree.xOff + (1.0 + float(leaf_nums)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(cntrpt, parent_pt, node_txt)
    plot_node(first_str, cntrpt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            plot_tree(second_dict[key], cntrpt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), cntrpt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntrpt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


def create_plot(intree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=True, **axprops)
    plot_tree.totalW = float(get_leafs_num(intree))
    plot_tree.totalD = float(get_tree_depth(intree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(intree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    my_tree = retrieve_tree(0)
    print(my_tree)
    print(get_leafs_num(my_tree))
    print(get_tree_depth(my_tree))
    # my_tree['no surfacing'][3] = 'maybe'
    create_plot(my_tree)


