#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: KALI LINUX
@编译器：python3.7
@Created on 2018-11-27 16:22
@Author:ChileWang
@algorithm：线性回归
"""
import matplotlib.pyplot as plt
from numpy import *


def load_dataset(file_name):
    num_feat = len(open(file_name).readline().split('\t')) - 1
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def stand_regress(x_arr, y_arr):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    xTx = x_mat.T * x_mat
    if linalg.det(xTx) == 0:  # 计算行列式
        print("This matrix is singula, cannot do inverse!")
        return
    ws = xTx.I * (x_mat.T * y_mat)
    return ws


def plot_regress(x_mat, y_mat, ws):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0], s=2, c='red')
    x_copy = x_mat.copy()
    x_copy.sort(0)  # 升序排序，避免画图混乱
    y_hat = x_copy * ws
    y_hat_prediction = x_mat * ws
    print(y_hat)
    corr = corrcoef(y_hat_prediction.T, y_mat)  # 相关系数, 该矩阵包含两两组合的系数
    print(corr)
    ax.plot(x_copy[:, 1], y_hat)
    plt.show()


def lwlr(test_point, x_arr, y_arr, k=1.0):
    """
    局部线性加权回归函数
    :param test_point:
    :param x_arr:
    :param y_arr:
    :param k:决定对附近的点赋予多大的权重
    :return:
    """
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    m = shape(x_mat)[0]
    weights = mat(eye(m))  # 创建对角矩阵
    for j in range(m):
        diff_mat = test_point - x_mat[j, :]
        weights[j, j] = exp(diff_mat * diff_mat.T / (-2.0 * k ** 2))  # 权值大小以指数级别衰减
    xTx = x_mat.T * (weights * x_mat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singula, cannot do inverse!")
        return
    ws = xTx.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def lwlr_test(test_arr, x_arr, y_arr, k=1.0):
    m = shape(test_arr)[0]
    y_hat = zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat


def plot_lwlr(x_arr, y_arr, y_hat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y_mat = mat(y_arr)
    x_mat = mat(x_arr)
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0], s=2, c='red')
    str_ind = x_mat[:, 1].argsort(0)  # 获取横坐标x的升序索引
    x_sort = x_mat
    x_sort.sort(0)  # 升序排序，避免画图混乱
    ax.plot(x_sort[:, 1], y_hat[str_ind])
    # ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T.flatten().A[0], s=2, c='red')
    plt.show()


def rss_error(y_arr, y_hat_arr):
    return ((y_arr - y_hat_arr) ** 2).sum()


def ridge_regres(x_mat, y_mat, lam=0.2):
    """
    岭回归
    :param x_mat:
    :param y_mat:
    :param lam:
    :return:回归系数
    """
    xTx = x_mat.T * x_mat
    denom = xTx + eye(shape(x_mat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singula, cannot do inverse!")
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws


def ridge_test(x_arr, y_arr):
    """
    得到30个λ的回归系数
    :param x_arr:
    :param y_arr:
    :return:
    """
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    # 数据标准化
    y_mean = mean(y_mat, 0)  # 按列求平均
    y_mat = y_mat - y_mean
    x_mean = mean(x_mat, 0)
    x_var = var(x_mat, 0)  # 按列求方差
    x_mat = (x_mat - x_mean) / x_var
    num_test_pts = 30
    w_mat = zeros((num_test_pts, shape(x_mat)[1]))
    for i in range(num_test_pts):
        ws = ridge_regres(x_mat, y_mat, exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat


def regularize(x_mat):  # regularize by columns
    """
    #数据标准化（特征标准化处理），减去均值，除以方差
    标准化数据通过减去均值然后除以方差（或标准差），
    这种数据标准化方法经过处理后数据符合标准正态分布，即均值为0，标准差为1
    :param x_mat:
    :return:
    """
    in_mat = x_mat.copy()
    in_means = mean(in_mat, 0)   # calc mean then subtract it off
    in_var = var(in_mat, 0)      # calc variance of Xi then divide by it
    in_mat = (in_mat - in_means) / in_var
    return in_mat


def stage_wise(x_arr, y_arr, eps, num_it=100):
    """
    前向逐步线性回归
    :param x_arr:
    :param y_arr:
    :param eps: 步长
    :param num_it: 迭代次数
    :return:
    """
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y_mean = mean(y_mat, 0)  # 按列求均值
    y_mat = y_mat - y_mean
    x_mat = regularize(x_mat)
    m, n = shape(x_mat)
    return_mat = zeros((num_it, n))
    ws = zeros((n, 1))
    ws_test = ws.copy()
    ws_max = ws.copy()
    """
    对每个特征的权值增加或者减少eps，看他们的影响，找到误差最小的w，并保存
    """
    for i in range(num_it):  # 迭代次数
        print(ws.T)
        lowerest_error = inf
        for j in range(n):  # 每个特征
            for sign in [-1, 1]:  # 增加或者减少
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = x_mat * ws_test
                rss_e = rss_error(y_mat.A, y_test.A)
                if rss_e < lowerest_error:
                    lowerest_error = rss_e
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T
    return return_mat


if __name__ == '__main__':
    # x_arr, y_arr = load_dataset('ex0.txt')
    # ws = stand_regress(x_arr, y_arr)
    # print(ws)
    # print(y_arr[0])
    # print(lwlr(x_arr[0], x_arr, y_arr, 1.0))
    # print(lwlr(x_arr[0], x_arr, y_arr, 0.001))
    # plot_regress(mat(x_arr), mat(y_arr), ws)
    # y_hat = lwlr_test(x_arr, x_arr, y_arr, 0.01)
    # plot_lwlr(x_arr, y_arr, y_hat)

    # ab_x, ab_y = load_dataset('abalone.txt')
    # y_hat01 = lwlr_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 0.1)
    # y_hat1 = lwlr_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 1)
    # y_hat10 = lwlr_test(ab_x[0:99], ab_x[0:99], ab_y[0:99], 10)
    # print('拟合')
    # print(rss_error(ab_y[0:99], y_hat01.T))
    # print(rss_error(ab_y[0:99], y_hat1.T))
    # print(rss_error(ab_y[0:99], y_hat10.T))
    # print('------------------')
    # print('预测')
    # print(rss_error(ab_y[100:199], lwlr_test(ab_x[100:199], ab_x[0:99], ab_y[0:99], 0.1).T))
    # print(rss_error(ab_y[100:199], lwlr_test(ab_x[100:199], ab_x[0:99], ab_y[0:99], 1).T))
    # print(rss_error(ab_y[100:199], lwlr_test(ab_x[100:199], ab_x[0:99], ab_y[0:99], 10).T))
    #
    # # 简单线性回归
    # print('----------')
    # print('简单线性回归')
    # ws = stand_regress(ab_x[0:99], ab_y[0:99])
    # y_hat = mat(ab_x[100:199]) * ws
    # print(rss_error(ab_y[100:199], y_hat.T.A))

    # 岭回归系数， 横轴log（λ）， 纵轴weights。
    # λ非常小的时候与普通线性回归一致，非常大的时候，weights归零，在他们中间可以找到最佳λ。
    ab_x, ab_y = load_dataset('abalone.txt')
    ridge_wights = ridge_test(ab_x, ab_y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_wights)
    plt.show()

    # 逐步线性回归
    ab_x, ab_y = load_dataset('abalone.txt')
    stage_wise(ab_x, ab_y, 0.01, 200)
    print('-----------')
    stage_wise(ab_x, ab_y, 0.001, 5000)


