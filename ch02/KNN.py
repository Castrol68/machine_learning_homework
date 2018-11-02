#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
编译器：python3.6
@Created on 2018-10-26 16:25
@Author:ChileWang
@algorithm：K-近邻算法
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def create_dataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):  # 核心代码 k-近邻算法
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 距离矩阵
    sqrdiffMat = diffMat ** 2  # 距离取平方
    sqDisctance = sqrdiffMat.sum(axis=1)  # 列相加
    sortedDistanceIndex = sqDisctance.argsort()  # argsort函数返回的是数组值从小到大的索引值,[3, 1, 2]从小到大为[1，2，3],期对应的索引为[1，2，0]
    classCount = dict()
    for i in range(k):
        voteIlabel = labels[sortedDistanceIndex[i]]  # 矩阵从小到大的索引对应的label值
        if voteIlabel in classCount.keys():  # 统计
            classCount[voteIlabel] = classCount[voteIlabel] + 1
        else:
            classCount[voteIlabel] = 1
    sorted_labels_dic = sorted(classCount.items(), key=lambda x: x[1])  # 字典按值倒序排序
    return sorted_labels_dic[-1][0]
    # distance_dic = dict()  # 记录已知类别数据集中的点与当前点的距离，以及当前数据集中的点属于哪个类别
    # for i in range(len(dataSet)):
    #     dataX = dataSet[i]
    #     distance = abs(dataX[0] - inX[0]) + abs(dataX[1] - inX[0])  # 计算已知类别数据集中的点与当前点的距离
    #     distance_dic[distance] = labels[i]
    #
    # sorted_distance_dic = sorted(distance_dic.items(), key=lambda x: x[0])  # 对字典排序
    # dis_labels = dict()
    # for i in range(k):  # 取前n个值，统计类别。
    #     if sorted_distance_dic[i][1] in dis_labels.keys():
    #         dis_labels[sorted_distance_dic[i][1]] += 1
    #     else:
    #         dis_labels[sorted_distance_dic[i][1]] = 1
    #
    # sorted_labels_dic = sorted(dis_labels.items(), key=lambda x: x[1])  # 对标签组排序
    #
    # label = sorted_labels_dic[-1][0]  # 统计值最大取最后一个，确定inX的标签并加入数据集
    # labels.append(label)
    # dataSet = np.insert(dataSet, len(dataSet), values=np.array([inX]), axis=0)
    # return dataSet, labels


def file2matrix(filename):
    labels = []
    dataSet = []
    with open(filename, 'r')as fr:
        for line in fr:
            line = line.rstrip()
            line = line.split('\t')
            for i in range(len(line)):
                line[i] = float(line[i])
            data = line[0:3]
            dataSet.append(data)
            labels.append(line[3])
    dataSet = np.array(dataSet)
    return dataSet, labels


def autoNorm(dataSet): #  归一化处理
    minValues = dataSet.min(0)  # 每列最小值
    maxValues = dataSet.max(0)  # 每列最大值
    ranges = maxValues - minValues
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minValues, (m, 1))  # 减去最小值
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # 除以范围，归一化
    return normDataSet, ranges, minValues


def datingClassTest():
    hoRatio = 0.1
    dataMatrix, labels = file2matrix('datingTestSet2.txt')
    normDataSet, ranges, minValues = autoNorm(dataMatrix)
    m = normDataSet.shape[0]
    numTestVe = int(m * hoRatio)  # 随机取来测试的数据总数
    errorCount = 0.0  # 错误率
    for i in range(numTestVe):
        classificerResult = classify(normDataSet[i, :], normDataSet[numTestVe:m, :], labels[numTestVe:m], 3)
        print('the classifier result is %d, the real result is %d' % (classificerResult, labels[i]))
        if classificerResult != labels[i]:
            errorCount += 1.0

    print('Error rate is %f' % (errorCount/float(numTestVe)))


def img2vetor(filename):
    vetor = []
    with open(filename, 'r') as fr:
        for line in fr:
            line = line.rstrip()
            for int_line in line:
                vetor.append(int(int_line))
    vetor = np.array(vetor)
    return vetor


def hand_writing_class_test():
    hwlabels = []
    train_mat = []
    train_file_list = os.listdir('trainingDigits')
    m = len(train_file_list)
    for i in range(m):
        files_name_str = train_file_list[i]
        file_str = files_name_str.split('.')[0]  # 获取不带后缀的文件名
        class_num_str = int(file_str.split('_')[0])  # 获取数字
        hwlabels.append(class_num_str)
        train_mat.append(img2vetor('trainingDigits/%s' % files_name_str))
    train_mat = np.array(train_mat)
    print(train_mat)

    test_file_list = os.listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        files_name_str = test_file_list[i]
        file_str = files_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vetor2test = img2vetor('testDigits/%s' % files_name_str)
        classfier_result = classify(vetor2test, train_mat, hwlabels, 3)
        print('the classifier result is %d, the real result is %d' % (classfier_result, class_num_str))
        if classfier_result != class_num_str:
            error_count += 1
    print('Error rate is %f' % (error_count / float(m_test)))


if __name__ == '__main__':
    # group, labels = create_dataset()
    # print(classify([1, 2], group, labels, 3))
    # dataMatrix, labels = file2matrix('datingTestSet2.txt')
    # autoNorm(dataMatrix)
    # print(dataMatrix)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)  # 一行一列，并画在第一个幕布
    # ax.scatter(dataMatrix[:, 1], dataMatrix[:, 2], 15.0 * np.array(labels), 15.0 * np.array(labels))
    # plt.show()
    # arr1 = np.arange(36)  # 创建一个一维数组。
    # arr2 = arr1.reshape(6, 6)  # 更改数组形状。
    # print(arr2)
    # print(arr2[0:3:, 0:3])  # 0~3行 0~3列
    # print(arr2**2)
    # arr = np.array([[1, 5], [4, 2]])
    # print(arr**2)
    # print(arr.sum(axis=1))
    #datingClassTest()
    hand_writing_class_test()



