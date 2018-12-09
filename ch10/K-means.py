#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: Ubuntu18.04
@编译器：python3.7
@Created on 2018-12-09 19:28
@Author:ChileWang
@algorithm:K-means
"""
from numpy import *


def load_dataset(file_name):
    data_mat = []
    with open(file_name, 'r') as fr:
        for line in fr:
            cur_line = line.strip().split('\t')
            flt_line = list(map(float, cur_line))
            data_mat.append(flt_line)
    return mat(data_mat)


def dist_eclud(vec_A, vec_B):
    """
    计算质心和数据点的距离
    :param vec_A:
    :param vecB:
    :return:
    """
    return sqrt(sum(power(vec_A - vec_B, 2)))


def rand_cent(dataset, k):
    """
    初始随机构建质心
    :param dataset:
    :param k:质心的个数
    :return:
    """
    n = shape(dataset)[1]
    centroid = mat(zeros((k, n)))
    for j in range(n):
        min_J = min(dataset[:, j])
        range_J = float(max(dataset[:, j]) - min_J)  # 特征值的范围
        centroid[:, j] = min_J + range_J * random.rand(k, 1)  # rand(row, col)
    return centroid


def k_means(dataset, k, dist_means=dist_eclud, create_cent=rand_cent):
    """
    :param dataset:
    :param k:
    :param dist_means: 距离计算函数
    :param create_cent: 生成质点函数
    :return:
    """
    m = shape(dataset)[0]
    cluster_assment = mat(zeros((m, 2)))  # 装距离质心最小的质心的索引与其距离(分配结果)
    centroid = create_cent(dataset, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = inf
            min_index = -1
            for j in range(k):  # 寻找最近的质心
                dist_JI = dist_means(centroid[j, :], dataset[i, :])
                if dist_JI < min_dist:
                    min_dist = dist_JI
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist**2
        # print(centroid)
        # print('------------')
        # 更新质心位置
        for cent in range(k):
            pts_in_clust = dataset[nonzero(cluster_assment[:, 0].A == cent)[0]]  # .A to array
            # 找到改变的质心的数据簇的所有数据
            centroid[cent, :] = mean(pts_in_clust, axis=0)  # 对这些数据按列求均值(更新质心的位置)
    return centroid, cluster_assment


def bi_KMeans(dataset, k,  distMeans=dist_eclud):
    """
    二分均值聚类
    :param dataset:
    :param k:
    :param distMeans:
    :return:
    """
    m = shape(dataset)[0]
    cluster_assiment = mat(zeros((m, 2)))
    centroids0 = mean(dataset, axis=0).tolist()[0]  # 初始看成一个簇聚类
    cent_list = [centroids0]
    for j in range(m):
        # 装距离质心最小的质心的索引与其距离(分配结果)
        cluster_assiment[j, 1] = distMeans(mat(centroids0), dataset[j, :])**2
    while len(cent_list) < k:
        lowest_SSE = inf  # 最小平方误差（距离）
        for i in range(len(cent_list)):
            pts_in_curr_cluster = dataset[nonzero(cluster_assiment[:, 0].A == i)[0], :]  # 抽取属于该簇的所有数据
            centroid_mat, split_cluster_ass = k_means(pts_in_curr_cluster, 2, distMeans)  # 开始二分聚类
            sse_split = sum(split_cluster_ass[:, 1])  # 计算二分聚类之后的距离误差
            # 计算不聚类的距离误差
            sse_not_split = sum(cluster_assiment[nonzero(cluster_assiment[:, 0].A != i)[0], 1])
            print('sse split and sse not split:', sse_split, sse_not_split)
            print('-------------------')
            if sse_split + sse_not_split < lowest_SSE:
                best_cent_spilt = i  # 最佳切分点
                best_new_cent = centroid_mat
                best_clus_ass = split_cluster_ass.copy()
                lowest_SSE = sse_not_split + sse_split
        # 更新簇的分配结果
        best_clus_ass[nonzero(best_clus_ass[:, 0].A == 1)[0], 0] = len(cent_list)  # 更新划分之后的簇编号，方便最终合并
        best_clus_ass[nonzero(best_clus_ass[:, 0].A == 0)[0], 0] = best_cent_spilt
        # print('the len of cluster assignment：', len(best_clus_ass))
        # print('the best split:', best_cent_spilt)
        cent_list[best_cent_spilt] = best_new_cent[0, :].tolist()[0]  # 更新质心
        cent_list.append(best_new_cent[1, :].tolist()[0])  # 新的质心加入
        cluster_assiment[nonzero(cluster_assiment[:, 0].A == best_cent_spilt)[0], :] = best_clus_ass
        # 属于原先被划分的簇的数据更改为划分后的编号
    return mat(cent_list), cluster_assiment


if __name__ == '__main__':
    # data_mat = load_dataset('testSet.txt')
    # cent = rand_cent(data_mat, 2)
    # print(cent)
    # print(dist_eclud(cent[0], cent[1]))
    # my_cend, assignment = k_means(data_mat, 4)
    # print(my_cend)
    data_mat = load_dataset('testSet2.txt')
    my_cend, assignment = bi_KMeans(data_mat, 3)
    print(my_cend)

