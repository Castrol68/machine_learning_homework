#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: KALI LINUX
@编译器：python3.7
@Created on 2018-11-08 19:47
@Author:ChileWang
@algorithm：决策树算法
"""
from math import log


def create_dataset():
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def calc_shannon_ent(dataset):  # 计算信息熵, 信息熵越大数据越混乱。
    numEntries = len(dataset)  # 所有实例
    labelcount = dict()
    for feaVec in dataset:
        current_label = feaVec[-1]  # 取最后一列的值
        if current_label not in labelcount.keys():
            labelcount[current_label] = 0
        labelcount[current_label] += 1
    shanno_ent = 0
    for key in labelcount.keys():
        prob = float(labelcount[key]) / numEntries
        shanno_ent -= prob * log(prob, 2)  # 计算信息熵
    return shanno_ent


def spilt_dataset(dataset, axis, value):  # 按照特定的特征划分数据集
    """
    :param dataset: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value:用以划分的特征值
    :return:返回已划分好的数据集
    """
    ret_dataset = []
    for feaVec in dataset:
        if feaVec[axis] == value:
            reduce_feat_vec = feaVec[:axis]
            reduce_feat_vec.extend(feaVec[axis + 1:])
            ret_dataset.append(reduce_feat_vec)

    return ret_dataset


def choose_best_feature_to_spilt(dataset):  # 找出信息增益最大的特征
    """
    :param dataset:
    :return: 最佳特征索引值
    """
    num_feature = len(dataset[0]) - 1  # 每个实例的特征数目
    base_entropy = calc_shannon_ent(dataset)  # 原始信息熵
    best_info_gain = 0.0  # 最佳信息增益
    best_feature_index = -1  # 最佳特征索引值
    for i in range(num_feature):
        feat_list = [example[i] for example in dataset]  # 取特征值，形成特征列表
        unique_val = set(feat_list)  # 确定唯一的分类标签（特征）
        new_entropy = 0.0

        for value in unique_val:  # 计算每种划分方式的信息熵
            sub_dataset = spilt_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy

        if info_gain > best_info_gain:  # 计算最佳增益
            best_info_gain = info_gain
            best_feature_index = i

    return best_feature_index


def majority_decision(class_list):  # 如果所有属性均遍历完，但类标签仍然不唯一，则采用多数表决法
    """
    :param class_list: 分类标签列表
    :return: 返回最终决定的分类标签
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sotred_class_count = sorted(class_count.items(), key=lambda x: x[1])  # 字典按值倒序排序
    return sotred_class_count[0][0]


def create_tree(dataset, labels):  # 递归构建决策树
    """
    :param dataset: 参与划分的数据集
    :param labels: 剩余的标签
    :return: 决策树
    """
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):  # 若类别标签完全相同，则停止继续划分
        return class_list[0]
    if len(dataset[0]) == 1:  # 遍历完所有特征，返回出现次数最多的
        return majority_decision(class_list)
    best_feat = choose_best_feature_to_spilt(dataset)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}  # 创建一个分类节点
    del(labels[best_feat])  # 删除已分类的标签
    feat_values = [example[best_feat] for example in dataset]  # 取最佳分类索引的特征值，形成列表
    unique_values = set(feat_values)
    for value in unique_values:
        sublabels = labels[:]
        my_tree[best_feat_label][value] = create_tree(spilt_dataset(dataset, best_feat, value), sublabels)

    return my_tree


if __name__ == '__main__':
    my_dataset, labels = create_dataset()
    print(my_dataset)
    print(calc_shannon_ent(my_dataset))
    print(spilt_dataset(my_dataset, 0, 1))
    print(spilt_dataset(my_dataset, 0, 0))
    print(create_tree(my_dataset, labels))

