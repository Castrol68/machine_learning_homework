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

    :param node_txt: 节点文本
    :param center_pt: 决策节点
    :param parent_pt:
    :param node_type:
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


if __name__ == '__main__':
    create_plot()

