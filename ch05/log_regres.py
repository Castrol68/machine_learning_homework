#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: KALI LINUX
@编译器：python3.7
@Created on 2018-11-13 09:49
@Author:ChileWang
@algorithm：Logistic回归梯度算法

小结
　　logistic回归的目的是寻找一个非线性函数sigmoid的最佳拟合参数，从而来相对准确的预测分类结果。
为了找出最佳的函数拟合参数，最常用的优化算法为梯度上升法，当然我们为了节省计算损耗，通常选择随机梯度上升法来迭代更新拟合参数。
并且，随机梯度上升法是一种在线学习算法，它可以在新数据到来时完成参数的更新，而不需要重新读取整个数据集来进行批处理运算。
　　总的来说，logistic回归算法，其具有计算代价不高，易于理解和实现等优点；
此外，logistic回归算法容易出现欠拟合，以及分类精度不太高的缺点。
"""
from numpy import *
import matplotlib.pyplot as plt
import random


def load_dataset():
    """
    　假设有100个样本点，每个样本有两个特征：x1和x2.此外为方便考虑，
    我们额外添加一个x0=1，将线性函数z=(wT)x+b转为z=(wT)x(此时向量w和x的维度均价1).
    可以理解为x0相当于 b

    那么梯度上升法的伪代码如下：

　　初始化每个回归系数为1
　　重复R次：
　　　　计算整个数据集梯度
　　　　使用alpha*gradient更新回归系数的向量
　　返回回归系数
    :return:
    """
    data_mat = []
    label_mat = []
    with open('testSet.txt', 'r') as fr:
        for line in fr:
            line_arr = line.strip().split()
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])  # 坐标（x，y），b=1.0
            label_mat.append(int(line_arr[2]))  # 分类标签

    return data_mat, label_mat


def sigmoid(inx):
    """
    :param inx:输入坐标（b, x1, y1)
    :return: sigmod函数
    """
    return 1.0 / (1 + exp(-inx))


def grad_ascent(data_mat, label_mat):
    """
    梯度上升算法
    :param data_mat:数据集
    :param label_mat:数据集对应的标签
    :return:回归系数
    """
    data_mat = mat(data_mat)  # 转换成矩阵
    label_mat = mat(label_mat).transpose()  # 转置
    # print(data_mat, label_mat)
    m, n = shape(data_mat)
    alpha = 0.001  # 移动步长
    max_cycle = 500  # 最大循环次数
    weights = ones((n, 1))  # 初始化回归系数，生成n行一列的矩阵
    # print(weights)
    for k in range(max_cycle):
        h = sigmoid(data_mat * weights)  # 计算误差
        error = label_mat - h
        """
        回归系数进行更新的公式为：w：w+alpha*gradient,其中gradient是对参数w求偏导数。
        则我们可以通过求导验证logistic回归函数对参数w的梯度为(yi-sigmoid(wTx))*x
        sigmoid(wT)x = h
        yi - h = error
        """
        weights = weights + alpha * data_mat.transpose() * error

    return weights


def plot_best_fit(weights):
    """
    :param wei:
    :return:
    """
    data_mat,  label_mat = load_dataset()
    data_arr = array(data_mat)
    n = shape(data_arr)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])  # 坐标（x,y)
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = array(arange(-3.0, 3.0, 1.0))  # 取六个点，步长为1
    """
    x = 0是（两个类别）sigmoid函数的分界线
    我们设定 0 = w0x0 + w1x1 + w2x2 此时x0 = 1
    y = x2 = -(w0 + w1x1)/w2
    """
    y = (-weights[0] - weights[1] * x) / weights[2]
    y = array(y)  # 随机梯度算法用的y
    # y = array(y)[0]  # 梯度上升算法用的y[0]
    # print(x)
    # print(y)
    print('b:', -weights[0]/weights[2])
    print('k:', -weights[1] / weights[2])
    print('k:', (y[0] - y[1])/(x[0] - x[1]))
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('Y1')
    plt.show()


def sto_grad_ascent0(data_mat, class_labels):
    """
    随机梯度上升算法
我们知道梯度上升法每次更新回归系数都需要遍历整个数据集，当样本数量较小时，该方法尚可，
但是当样本数据集非常大且特征非常多时，那么随机梯度下降法的计算复杂度就会特别高。
一种改进的方法是一次仅用一个样本点来更新回归系数，即随机梯度上升法。
由于可以在新样本到来时对分类器进行增量式更新，因此随机梯度上升法是一个在线学习算法。

随机梯度上升法可以写成如下伪代码：

所有回归系数初始化为1
对数据集每个样本
　　计算该样本的梯度
　　使用alpha*gradient更新回顾系数值
 返回回归系数值

    :param data_mat:
    :param class_labels:
    :return:
    """
    m, n = shape(data_mat)
    alpha = 0.01  # 步长
    weights = ones(n)  # 回归系数
    for i in range(m):
        h = sigmoid(sum(data_mat[i] * weights))  # 点乘即对应相乘
        error = class_labels[i] - h
        # print(class_labels[i])
        # print(error)
        weights = weights + alpha * error * data_mat[i]
        # print(weights)
        # print('---------------')
    return weights


def promoved_sto_grad_ascent0(data_mat, class_labels, num_iter=150):
    """
    改进的随机梯度上升算法
    :param data_mat:数据集
    :param class_labels:数据集对应的标签
    :param num_iter:迭代次数， 默认150
    :return:回归系数
    """
    m, n = shape(data_mat)
    weights = ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 保证alpha每次迭代进行调整， 避免数据波动或者高频波动
            ran_index = int(random.uniform(0, len(data_index)))  # 随机选取样本进行更新回归系数，减少周期性波动
            h = sigmoid(sum(data_mat[ran_index] * weights))  # 点乘
            error = class_labels[ran_index] - h
            weights = weights + alpha * error * data_mat[ran_index]
            # print(error)
            # print(weights)
            # print('------------')
            del(data_index[ran_index])

    return weights


def classify_vector(inx, weights):
    """
    :param inx: 输入
    :param weights: 回归系数
    :return: 返回类别
    """
    prob = sigmoid(sum(inx * weights))
    if prob > 0.5:
        return 1.0
    return 0.0


def colic_test():
    training_set = []
    training_label = []
    #  训练
    with open('horseColicTraining.txt', 'r') as fr:
        for line in fr:
            curr_line = line.strip().split('\t')
            line_arr = []
            for i in range(21):
                line_arr.append(float(curr_line[i]))
            training_set.append(line_arr)
            training_label.append(float(curr_line[21]))
    training_wei = promoved_sto_grad_ascent0(array(training_set), training_label, 500)  # 回归系数

    # 测试
    error_count = 0
    num_test_vec = 0.0
    with open('horseColicTest.txt', 'r') as fr:
        for line in fr:
            num_test_vec += 1.0
            curr_line = line.strip().split('\t')
            line_arr = []
            for i in range(21):
                line_arr.append(float(curr_line[i]))
            if int(classify_vector(array(line_arr), training_wei)) != int(curr_line[21]):
                error_count += 1
    error_rate = float((error_count) / num_test_vec)
    print('the error rate of this test is :%f' % error_rate)
    return error_rate


def multi_test():   # 多次测试求平均值
    num_test = 10
    error_sum = 0.0
    for k in range(num_test):
        error_sum += colic_test()
    print('after %d iteration the average error rate is %f' % (num_test, error_sum/float(num_test)))


if __name__ == '__main__':
    data_mat, label_mat = load_dataset()
    # print(data_mat, label_mat)
    weights = grad_ascent(data_mat, label_mat)
    # print(weights)
    # weights = sto_grad_ascent0(array(data_mat), label_mat)  # 都是array数组
    # weights = promoved_sto_grad_ascent0(array(data_mat), label_mat)  # 都是array数组
    # print(weights)
    print(weights)
    plot_best_fit(weights)
    # multi_test()

