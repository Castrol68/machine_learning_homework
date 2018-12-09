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
from tkinter import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


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
            new_s = err_type(mat0) + err_type(mat1)  # 切完算总方差
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
    ret_tree['spind'] = feat  # 最佳切分特征的索引
    ret_tree['spval'] = val  # 最佳切分特征值
    l_set, r_set = bin_split_dataset(dataset, feat, val)
    ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(r_set, leaf_type, err_type, ops)
    return ret_tree


def is_tree(obj):
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    """
    计算所有叶子节点的平均值
    :param tree:
    :return:
    """
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['right'] + tree['left']) / 2.0


def prune(tree, testdada):
    """
    树剪枝
    :param tree: 待剪枝的树
    :param testdada:剪枝所需的测试数据
    :return:
    """
    if shape(testdada)[0] == 0:
        return get_mean(tree)  # 没有测试数据则对树进行塌陷处理

    """
    假设训练集构建的树过拟合，
    利用非空的测试集合对其进行剪枝
    """
    if is_tree(tree['right']) or is_tree(tree['left']):
        l_set, r_set = bin_split_dataset(testdada, tree['spind'], tree['spval'])
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], r_set)
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], l_set)
    if (not is_tree(tree['right'])) and (not is_tree(tree['left'])):
        # 若剪枝后均为叶子节点，比较两者合并前后的误差，选误差最小的返回
        l_set, r_set = bin_split_dataset(testdada, tree['spind'], tree['spval'])
        # power():数组元素求n次方
        error_no_merge = sum(power(l_set[:, -1] - tree['left'], 2)) + sum(power(r_set[:, -1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right']) / 2.0

        error_merge = sum(power(testdada[:, -1] - tree_mean, 2))
        if error_merge < error_no_merge:
            # print('merging!')
            return tree_mean
        else:
            return tree
    else:
        return tree


def liner_solve(dataset):
    """
    将数据集格式化为X， Y
    :param dataset:
    :return:
    """
    m, n = shape(dataset)
    X = mat(ones((m, n)))  # 训练的X值
    Y = mat(ones((m, 1)))  # 训练的Y值
    X[:, 1:n] = dataset[:, 0: n-1]  # 默认截距为1
    Y = dataset[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix can not do inverse!')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def model_leaf(dataset):
    """
    模型叶子
    :param dataset:
    :return:
    """
    ws, X, Y = liner_solve(dataset)
    return ws


def model_err(dataset):
    """
    计算误差
    :param dataset:
    :return:
    """
    ws, X, Y = liner_solve(dataset)
    y_hat = X * ws
    return sum(power(Y - y_hat, 2))


def reg_tree_eval(model, in_data):
    """
    :param model:
    :param in_data:
    :return:
    """
    return float(model)


def model_tree_eval(model, in_data):
    """
    :param model:
    :param in_data:
    :return:
    """
    n = shape(in_data)[1]
    x = mat(ones((1, n + 1)))
    x[:, 1:n+1] = in_data
    return float(x * model)


def tree_forecast(tree, indata, model_eval=reg_tree_eval):
    """
    对给定的树结构，输入一个数据点或者行向量， 返回其预测值
    :param tree:
    :param indata:
    :param model_eval:
    :return:
    """
    if not is_tree(tree):
        return model_eval(tree, indata)
    if indata[tree['spind']] > tree['spval']:
        if is_tree(tree['left']):
            return tree_forecast(tree['left'], indata, model_eval)
        else:
            return model_eval(tree['left'], indata)
    else:
        if is_tree(tree['right']):
            return tree_forecast(tree['right'], indata, model_eval)
        else:
            return model_eval(tree['right'], indata)


def creat_forcast(tree, test_data, model_eval = reg_tree_eval):
    """
    多次调用预测函数
    :param tree:
    :param test_data:
    :param model_eval:
    :return:
    """
    m = len(test_data)
    y_hat = mat(zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_forecast(tree, mat(test_data[i]), model_eval)
    return y_hat


# def re_draw(tols, toln):
#     re_draw.f.clf()
#     re_draw.a = re_draw.f.add_subplot(111)
#     if chkBtnVar.get():
#         if toln < 2:
#             toln = 2
#         my_tree = create_tree(re_draw.rawDat, model_leaf, model_err, (tols, toln))
#         y_hat = creat_forcast(my_tree, re_draw.testDat, model_tree_eval)
#     else:
#         my_tree = create_tree(re_draw.rawDat, ops=(tols, toln))
#         y_hat = creat_forcast(my_tree, re_draw.testDat)
#     re_draw.a.scatter(re_draw.rawDat[:, 0], re_draw.rawDat[:, 1], s=5)
#     re_draw.a.plot(re_draw.testDat, y_hat, linewidth=2.0)
#     re_draw.canvas.show()

#
# def getInputs():
#     try:
#         tolN = int(tolN_entry.get())
#     except:
#         tolN = 10
#         print("enter Integer for tolN")
#         tolN_entry.delete(0, END)
#         tolN_entry.insert(0, '10')
#     try:
#         tolS = float(tolS_entry.get())
#     except:
#         tolS = 1.0
#         print("enter Float for tolS")
#         tolS_entry.delete(0, END)
#         tolS_entry.insert(0, '1.0')
#     return tolN, tolS
#
#
# def draw_new_tree():
#     tolN, tolS = getInputs()  # get values from Entry boxes
#     re_draw(tolS, tolN)
#
#
# def gui():
#     root = Tk()
#     Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)
#     Label(root, text="tolN").grid(row=1, column=0)
#     tolN_entry = Entry(root)
#     tolN_entry.grid(row=1, column=1)
#     tolN_entry.insert(0, '10')
#     Label(root, text='tolS').grid(row=1, column=0)
#     tolS_entry = Entry(root)
#     tolS_entry.grid(row=2, column=1)
#     tolS_entry.insert(0, '1.0')
#     Button(root, text='Redraw', command=draw_new_tree).grid(row=1, column=2, rowspan=3)
#
#     chkBtnvar = IntVar()
#     chkBtn = Checkbutton(root, text='Model Tree', variable=chkBtnvar)
#     chkBtn.grid(row=3, column=0, columnspan=2)
#
#     re_draw.rawDat = mat(load_dataset('sine.txt'))
#     re_draw.testDat = arange(min(re_draw.rawDat[:, 0]), max(re_draw.rawDat[:, 0]), 0.01)
#     re_draw(1.0, 10)
#     root.mainloop()


if __name__ == '__main__':
    # test_mat = mat(eye(4))
    # print(test_mat[:, -1])
    # mat0, mat1 = bin_split_dataset(test_mat, 1, 0.5)
    # print(mat0)
    # print(mat1)
    # my_mat = load_dataset('ex00.txt')
    # my_mat = mat(my_mat)
    # tree = create_tree(my_mat)
    # print(tree)
    # my_mat = load_dataset('ex0.txt')
    # my_mat = mat(my_mat)
    # tree = create_tree(my_mat)
    # print(tree)
    # get_mean(tree)
    # my_mat = load_dataset('ex2.txt')
    # my_mat = mat(my_mat)
    # tree = create_tree(my_mat, ops=[0, 1])
    # print(tree)
    # my_data_test = load_dataset('ex2test.txt')
    # my_data_test = mat(my_data_test)
    # tree = prune(tree, my_data_test)
    # print(tree)
    # my_mat = mat(load_dataset('exp2.txt'))
    # tree = create_tree(my_mat, model_leaf, model_err, (1, 10))
    # print(tree)
    #
    training_mat = mat(load_dataset('bikeSpeedVsIq_train.txt'))
    test_mat = mat(load_dataset('bikeSpeedVsIq_test.txt'))
    my_tree = create_tree(training_mat, ops=[0, 20])
    y_hat = creat_forcast(my_tree, test_mat[:, 0])
    cor = corrcoef(y_hat, test_mat[:, 1], rowvar=0)[0, 1]
    print(cor)

    model_tree = create_tree(training_mat, model_leaf, model_err, (1, 20))
    y_hat = creat_forcast(model_tree, test_mat[:, 0], model_tree_eval)
    cor = corrcoef(y_hat, test_mat[:, 1], rowvar=0)[0, 1]
    print(cor)
    # gui()


