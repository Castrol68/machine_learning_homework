#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: Ubuntu18.04
@编译环境：python3.7
@Created on 2018-12-17 21:33
@Author:ChileWang
@algorithm：Principal Component Analysis
将数据转换成前N个主成分的伪代码
    去除平均值
    计算协方差矩阵
    计算协方差矩阵的特征值和特征向量
    将特征值由大到小排列
    保留最上面的N个特征
    将数据转换到上述N个特征向量构建的新空间中
"""
from numpy import *
import matplotlib.pyplot as plt


def load_data_set(file_name, de_lim='\t'):
    fr = open(file_name)
    string_arr = [line.strip().split(de_lim) for line in fr.readlines()]
    data_arr = [list(map(float, line)) for line in string_arr]
    print(data_arr)
    return mat(data_arr)


def pca(data_mat, top_N_feat=999999):
    """
    :param data_mat:
    :param top_N_feat: 确定特征数目。 可以先计算前几个的方差总占比，从而确定特征数目。
    :return:
    """
    mean_val = mean(data_mat, axis=0)  # 按列取平均值
    mean_removed = data_mat - mean_val  # 去平均值
    # print(shape(mean_val))
    cov_mat = cov(mean_removed, rowvar=0)
    eig_vals, eig_vects = linalg.eig(mat(cov_mat))  # 特征值与特征向量
    eig_val_index = argsort(eig_vals)  # 从小到大排序
    eig_val_index = eig_val_index[:-(top_N_feat + 1): -1]  # 从大到小排序
    red_eig_vects = eig_vects[:, eig_val_index]
    # print(shape(mean_removed))
    # print(shape(red_eig_vects))
    # print(red_eig_vects)
    low_D_data_mat = mean_removed * red_eig_vects  # 转换到新空间
    print(shape(low_D_data_mat))
    recon_mat = (low_D_data_mat * red_eig_vects.T) + mean_val  # 重构原来的数据集用于调试
    print(shape(recon_mat))
    return low_D_data_mat, recon_mat


def draw_plot(data_mat, recon_mat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # .flatten().A 矩阵转化成array
    ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(recon_mat[:, 0].flatten().A[0], recon_mat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()


def replace_nan_with_mean():
    """
    用平均值替换缺失值NAN
    :return:
    """
    data_mat = load_data_set('secom.data', ' ')
    num_feat = shape(data_mat)[1]
    for i in range(num_feat):
        mean_val = mean(data_mat[nonzero(~isnan(data_mat[:, i].A))[0], i])  # 计算所有非nan的平均值
        data_mat[nonzero(isnan(data_mat[:, i].A))[0], i] = mean_val  # 替换成平均值
    return data_mat


if __name__ == '__main__':
    # a = [[1, 2, 3], [-2, 3, 9]]
    # a = array(a)
    # a = mat(a)
    # mean_val = mean(a, axis=0)
    # print(a)
    # print(mean_val)
    # print(a - mean_val)
    data_mat = load_data_set('testSet.txt')
    low_D_mat, recon_mat = pca(data_mat, 1)
    draw_plot(data_mat, recon_mat)
    data_mat = replace_nan_with_mean()