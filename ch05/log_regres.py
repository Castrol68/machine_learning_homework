#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: KALI LINUX
@编译器：python3.7
@Created on 2018-11-13 09:49
@Author:ChileWang
@algorithm：Logistic回归梯度算法
"""
from numpy import *
import matplotlib.pyplot as plt
import random


def load_dataset():
    data_mat = []
    label_mat = []
    with open('testSet.txt', 'r') as fr:
        for line in fr:
            line_arr = line.strip().split()
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])  # 回归系数，坐标（x，y）
            label_mat.append(int(line_arr[2]))  # 分类标签

    return data_mat, label_mat


def sigmoid(inx):
    return 1.0 / (1 + exp(-inx))


def grad_ascent(data_mat, label_mat):
    """
    梯度上升算法
    :param data_mat:
    :param label_mat:
    :return:
    """
    data_mat = mat(data_mat)  # 转换成矩阵
    label_mat = mat(label_mat).transpose()  # 转置
    # print(data_mat, label_mat)
    m, n = shape(data_mat)
    alpha = 0.001
    max_cycle = 500  # 最大循环次数
    weights = ones((n, 1))  # 生成n行一列的矩阵
    # print(weights)
    for k in range(max_cycle):
        h = sigmoid(data_mat * weights)
        error = label_mat - h
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
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = array(arange(-3.0, 3.0, 1.0))  # 取六个点，步长为1
    y = (-weights[0] - weights[1] * x) / weights[2]
    y = array(y)  # 随机梯度算法用的y
    # y = array(y)[0] # 梯度上升算法用的y
    # print(x)
    # print(y)
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('Y1')
    plt.show()


def sto_grad_ascent0(data_mat, class_labels):
    """
    随机梯度上升算法
    :param data_mat:
    :param class_labels:
    :return:
    """
    m, n = shape(data_mat)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(data_mat[i] * weights))  # 点乘
        error = class_labels[i] - h
        weights = weights + alpha * error * data_mat[i]
    return weights


def promoved_sto_grad_ascent0(data_mat, class_labels, num_iter=150):
    """
    改进的随机梯度上升算法
    :param data_mat:
    :param class_labels:
    :param num_iter:
    :return:
    """
    m, n = shape(data_mat)
    weights = ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            ran_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_mat[ran_index] * weights))  # 点乘
            error = class_labels[ran_index] - h
            weights = weights + alpha * error * data_mat[ran_index]
            del(data_index[ran_index])

    return weights


def classify_vector(inx, weights):
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
    training_wei = promoved_sto_grad_ascent0(array(training_set), training_label, 500)

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
    # weights = grad_ascent(data_mat, label_mat)
    # print(weights)
    # weights = sto_grad_ascent0(array(data_mat), label_mat)  # 都是array数组
    # weights = promoved_sto_grad_ascent0(array(data_mat), label_mat)  # 都是array数组
    # print(weights)
    # plot_best_fit(weights)
    multi_test()

