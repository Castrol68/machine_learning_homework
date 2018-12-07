#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: KALI LINUX
@编译器：python3.7
@Created on 2018-11-27 16:22
@Author:ChileWang
@algorithm：树回归
"""

from numpy import *


def load_dataset(file_name):
    data_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = list(map(float, cur_line))  # 将每行映射成浮点数
        data_mat.append(flt_line)
    return data_mat


def bin_split_dataset(dataset, feature, value):
    """
    按照指定的特征切分数据集
    :param dataset:
    :param feature:
    :param value:
    :return:
    """
    mat0 = dataset[nonzero(dataset[:, feature] > value)[0], :]
    mat1 = dataset[nonzero(dataset[:, feature] <= value)[0], :]
    # print(nonzero(dataset[:, feature] <= value))  # 0是nonzero行维度描述
    # print(nonzero(dataset[:, feature] <= value))  # 1是nonzero列维度描述
    return mat0, mat1


def reg_leaf(dataset):
    """
    均值, 当特征不再切分时，生成叶节点
    :param dataset:
    :return:
    """
    return mean(dataset[:, -1])


def reg_err(dataset):
    """
    总方差
    :param dataset:
    :return:
    """
    return var(dataset[:, -1]) * shape(dataset)[0]


def choose_best_split(dataset, leaf_type, err_type, ops):
    """
    对每个特征
        对每个特征值
            将数据集分成两份
            计算切分误差
            如果当前误差小于最小误差，替代当前最小误差
    :param dataset:
    :param leaf_type: 构造叶子节点
    :param err_type: 方差函数
    :param ops:
    :return:
    """
    tol_s = ops[0]  # 差错容许的下降值， 若小于此，则退出。
    tol_n = ops[1]  # 切分容许的最小样本数， 若小于此，则退出。

    if len(set(dataset[:, -1].T.tolist()[0])) == 1:  # 所有值相等则退出
        return None, leaf_type(dataset)
    m, n = shape(dataset)
    s = err_type(dataset)  # 切分误差
    best_s = inf  # 最佳切分误差
    best_index = 0
    best_value = 0
    for feat_index in range(n - 1):
        for split_val in set(dataset[:, feat_index].T.tolist()[0]):
            mat0, mat1 = bin_split_dataset(dataset, feat_index, split_val)
            if (shape(mat0)[0] < tol_n) or (shape(mat1)[0] < tol_n):
                continue
            new_s = err_type(mat0) + err_type(mat1)
            if new_s < best_s:
                best_index = feat_index
                best_value = split_val
                best_s = new_s
    if s - best_s < tol_s:  # 若误差减小不大则退出
        return None, leaf_type(dataset)
    mat0, mat1 = bin_split_dataset(dataset, best_index, best_value)
    if (shape(mat0)[0] < tol_n) or (shape(mat1)[0] < tol_n):  # 若切分出来的数据集很小，则退出
        return None, leaf_type(dataset)
    return best_index, best_value


def create_tree(dataset, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    """
    :param dataset:
    :param leaf_type: 建立叶子节点函数
    :param err_type: 误差计算函数
    :param ops:包含树构建所需的其他元素的元组
    :return:
    """
    feat, val = choose_best_split(dataset, leaf_type, err_type, ops)
    if feat == None:  # 满足停止条件则停止
        return val
    ret_tree = {}
    ret_tree['spind'] = feat
    ret_tree['spval'] = val
    l_set, r_set = bin_split_dataset(dataset, feat, val)
    ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(r_set, leaf_type, err_type, ops)
    return ret_tree


if __name__ == '__main__':
    # test_mat = mat(eye(4))
    # print(test_mat[:, -1])
    # mat0, mat1 = bin_split_dataset(test_mat, 1, 0.5)
    # print(mat0)
    # print(mat1)
    my_mat = load_dataset('ex00.txt')
    my_mat = mat(my_mat)
    tree = create_tree(my_mat)
    print(tree)
