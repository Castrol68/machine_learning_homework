#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: KALI LINUX
@编译器：python3.7
@Created on 2018-11-16 10:05
@Author:ChileWang
@algorithm：SVM
"""
from numpy import *
from sklearn.svm import SVC


def load_dataset(filename):
    """
    :param filename:
    :return: （数据点集，标签集）
    """
    data_mat = []
    label_mat = []
    with open(filename, 'r') as fr:
        for line in fr:
            line_arr = line.split('\t')
            data_mat.append([float(line_arr[0]), float(line_arr[1])])
            label_mat.append(float(line_arr[2]))

    return data_mat, label_mat


def selecr_jrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, h, l):
    if aj > h:
        aj = h
    if aj < l:
        aj = l
    return aj


def smo_simple(data_mat_in, class_labels, c, toler, max_iter):
    """
    简化版smo算法
    :param data_mat_in:数据集
    :param class_labels:类别标签
    :param c:常数C，惩罚因子
    :param toler:容错率
    :param max_iter:最大循环次数
    :return:
    """
    data_mat = mat(data_mat_in)  # 矩阵
    label_mat = mat(class_labels).transpose()  # 转置
    b = 0
    m, n = shape(data_mat)
    alphas = mat(zeros(m, 1))
    iter = 0
    while iter < max_iter:
        alpha_paris_changed = 0  # 用于记录alpha是否已经被优化
        for i in range(m):
            # 求g（xi） = ∑j～N(aj * yj K(xi, xj) + b) 找到训练集中满足KKT条件的点， 即yi * g（xi） = 1， 即找到0<ai<c
            fxi = float(multiply(alphas, label_mat).T * (data_mat * data_mat[i, :].T)) + b
            ei = fxi - float(label_mat[i])
            if (label_mat[i] * ei < -toler and alphas[i] < c) or \
                (label_mat[i] * ei > toler and alphas[i] > 0):
                # 找a2使其变化足够大
                j = selecr_jrand(i, m)
                fxj = float(multiply(alphas, label_mat).T * (data_mat * data_mat[j, :].T)) + b
                ej = fxj - float(label_mat[j])
                alphas_iold = alphas[i].copy()
                alphas_jold = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    l = max(0, alphas[j] - alphas[i])
                    h = min(c, c + alphas[j] - alphas[i])
                else:
                    l = max(0, alphas[i] + alphas[j] - c)
                    h = min(c, alphas[i] + alphas[j])
                if l == h:  # 若不能是a2变化的足够大，则放弃
                    print('L == H')
                    continue
                eta = 2.0 * data_mat[i, :] * data_mat[j, :].T - \
                    data_mat[i, :] * data_mat[i, :].T - \
                    data_mat[j, :] * data_mat[j, :].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alphas[j] -= label_mat[j] * (ei - ej) / eta
                alphas[j] = class_labels(alphas[j], h, l)
                if abs(alphas[j] - alphas_jold) < 0.00001:
                    print('j not moving enough')
                    continue
                alphas[i] += label_mat[j] * label_mat[i] * (alphas_jold - alphas[j])

                b1 = b - ei - label_mat[i] * (alphas[i] - alphas_iold) * \
                    data_mat[i, :] * data_mat[i, :].T - \
                    label_mat[j] * (alphas[j] - alphas_jold) * \
                    data_mat[i, :] * data_mat[j, :].T

                b2 = b - ei - label_mat[i] * (alphas[i] - alphas_iold) * \
                     data_mat[i, :] * data_mat[j, :].T - \
                     label_mat[j] * (alphas[j] - alphas_jold) * \
                     data_mat[j, :] * data_mat[j, :].T

                if (0 < alphas[i]) and (c > alphas[j]):
                    b = b1
                elif (0 < alphas[j]) and (c > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_paris_changed += 1
                print('iter: %d, i:%d, pairs changed %d ' % (iter, i, alpha_paris_changed))
        if alpha_paris_changed == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number:%d" % iter)
    return b, alphas


class opt_struct():
    def __init__(self, data_mat_in, class_labels, c, toler):
        """
        :param data_mat_in: 数据集
        :param class_labels: 分类标签
        :param c: 惩罚因子
        :param toler: 容忍错误率ξ
        """
        self.X = data_mat_in
        self.labels = class_labels
        self.c = c
        self.tol = toler
        self.m = shape(data_mat_in)[0]  # 行数
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.e_cache = mat(zeros((self.m, 2)))  # 误差缓存 第一列为是否有效标志位， 第二列是实际的e值


def calc_ek(os, k):
    """
    fxk = ∑（α * y * x * xi) + b
    :param os:
    :param k:
    :return:计算预测值与真实值的误差
    """
    fxk = float(multiply(os.alphas, os.labels).T * (os.X * os.X[k, :].T)) + os.b
    ek = fxk - float(os.labels[k])
    return ek


def select_J(i, os, ei):
    """
    选择第二个α
    :param i:
    :param os:
    :param ei:
    :return:
    """
    max_k = -1  # 最αi相对应的第二个αj的索引
    max_delta_e = 0
    ej = 0
    valid_ecache_list = nonzero(os.e_cache[:, 0].A)[0]
    # 返回的是非零e值所对应的索引,以此可知其对应的α .A 代表将 矩阵转化为array数组类型
    if len(valid_ecache_list) > 1:
        for k in valid_ecache_list:
            if k == i:
                continue
            ek = calc_ek(os, k)
            del_e = abs(ei - ek)
            if del_e > max_delta_e:  # 找到变化最大的α
                max_k = k
                max_delta_e = del_e
                ej = ek
        return max_k, ej
    else:  # 找不到则用以下函数赋值一个
        j = selecr_jrand(i, os.m)
        ej = calc_ek(os, j)
    return j, ej


def update_ek(os, k):
    """
    保存误差
    :param os:
    :param k:
    :return:
    """
    ek = calc_ek(os, k)
    os.e_cache[k] = [1, ek]


def inner_loop(i, os):
    """
    smo内层循环， 选择与αi变化最大的αj
    :param i:
    :param os:
    :return:
    """
    ei = calc_ek(os, i)
    if (os.labels[i] * ei < -os.tol and os.alphas[i] < os.c) or \
        (os.labels[i] * ei > os.tol and os.alphas[i] > 0): # 找到违反kkt条件最严重的点
        j, ej = select_J(i, os, ei)
        alphaI_old = os.alphas[i].copy()  # αi_old
        alphaJ_old = os.alphas[j].copy()  # αj_old

        if os.labels[i] != os.labels[j]:  # 二次规划求出两个端点
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.c, os.c + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.c)
            H = min(os.c, os.alphas[j] + os.alphas[i])

        if L == H:
            print('L==H')
            return 0
        eta = 2.0 * os.X[i, :] * os.X[j, :].T - os.X[i, :] * os.X[i, :].T - os.X[j, :] * os.X[j, :].T
        # ETA = -(xi^2 + xj ^2 - 2xixj)
        if eta >= 0:
            print('eta >= 0')
            return 0
        os.alphas[j] -= os.labels[j] * (ei - ej) / eta
        os.alphas[j] = clip_alpha(os.alphas[j], H, L)  # 算出αj
        update_ek(os, j)  # 更新误差缓存
        if os.alphas[j] - alphaJ_old < 0.00001:
            print('j not moving enough!')
            return 0
        os.alphas[i] += os.labels[j] * os.labels[i] * (alphaJ_old - os.alphas[j])   # 算出αi
        update_ek(os, i)  # 更新误差缓存

        # 算b1, b2 并且保证b1, b1 均大于零小于C, 否则b = (b1 + b2) / 2
        b1 = os.b - ei - os.labels[i] * (os.alphas[i] - alphaI_old) * os.X[i, :] * os.X[i, :].T - os.labels[j] * \
             (os.alphas[j] - alphaJ_old) * os.X[i, :] * os.X[j, :].T

        b2 = os.b - ej - os.labels[i] * (os.alphas[i] - alphaI_old) * os.X[j, :] * os.X[i, :].T - os.labels[j] * \
             (os.alphas[j] - alphaJ_old) * os.X[j, :] * os.X[j, :].T

        if (0 < os.alphas[i]) and (os.c > os.alphas[i]):
            os.b = b1
        elif (0 < os.alphas[j]) and (os.c > os.alphas[j]):
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smo_P(data_mat_in, class_labels, c, toler, max_iter, kTup = ('lin', 0)):
    os = opt_struct(mat(data_mat_in), mat(class_labels).transpose(), c, toler)
    iter = 0
    entre_set = True
    alpha_pairs_changed = 0
    # 交替遍历
    while iter < max_iter and (alpha_pairs_changed > 0 or entre_set):
        # 当遍历整个α集都未对α进行修改时则退出循环
        alpha_pairs_changed = 0
        if entre_set:  # 遍历所有值  遍历任意可能的α
            for i in range(os.m):
                alpha_pairs_changed += inner_loop(i, os)
                print('fullset, iter: %d i:%d ,paris changed:%d' % (iter, i, alpha_pairs_changed))
            iter += 1
        else:  # 检查非边界值 遍历不在边界的α
            non_boundIs = nonzero((os.alphas.A > 0) * (os.alphas.A < os.c))[0]
            for i in non_boundIs:
                alpha_pairs_changed += inner_loop(i, os)
                print('non-bound, iter: %d i:%d ,paris changed:%d' % (iter, i, alpha_pairs_changed))
            iter += 1
        if entre_set:
            entre_set = False
        elif alpha_pairs_changed == 0:
            entre_set = True
        print("iteration number: %d" % iter)
    return os.b, os.alphas


def calc_ws(alphas, data_arr, class_labels):
    x = mat(data_arr)
    label_mat = mat(class_labels).transpose()
    m, n = shape(x)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * label_mat[i], x[i, :].T)
    return w


if __name__ == '__main__':
    data_arr, labels = load_dataset('testSet.txt')
    b, alphas = smo_P(data_arr, labels, 0.6, 0.001, 40)
    ws = calc_ws(alphas, data_arr, labels)
    # print(ws)
    data_mat = mat(data_arr)
    print(data_mat[0])
    print(labels[0])
    print(data_mat[0] * mat(ws) + b)
    print('-------------------')
    clf = SVC()
    clf.fit(data_arr, labels)
    print(clf.predict(data_mat[0]))
