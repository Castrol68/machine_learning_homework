#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: KALI LINUX
@编译器：python3.7
@Created on 2018-11-26 21:32
@Author:ChileWang
@algorithm：adaboot
"""
from numpy import *
import matplotlib.pyplot as plt
import turtle
import random


def load_simple_data():
    data_mat = matrix([[1.,  2.1],
                     [2.,  1.1],
                     [1.3,  1.],
                     [1.,  1.],
                     [2.,  1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


def stump_classify(data_mat, dimen, thresh_val, thresh_ineq):
    """
    通过阈值比较对数据进行分类
    :param data_mat:
    :param dimen:特征维度
    :param thresh_val:阈值
    :param thresh_ineq:大于号或者小于号
    :return:
    """
    ret_arr = ones((shape(data_mat)[0], 1))
    if thresh_ineq == 'lt':
        ret_arr[data_mat[:, dimen] <= thresh_val] = -1.0
    else:
        ret_arr[data_mat[:, dimen] > thresh_val] = -1.0
    return ret_arr


def bulid_stump(data_arr, class_label, D):
    """
    单层决策树生成函数
    :param data_arr:
    :param class_label:
    :param D:
    :return:最佳单层决策树
    """
    data_mat = mat(data_arr)
    label_mat = mat(class_label).T
    m, n = shape(data_mat)
    num_steps = 10.0
    best_stump = {}
    best_clas_est = mat(zeros((m, 1)))
    min_err = inf  # 初始化为无穷
    for i in range(n):  # 遍历所有特征
        range_min = data_mat[:, i].min()
        rabge_max = data_mat[:, i].max()
        step_size = (rabge_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):  # 遍历每个步长
            for inequal in ['lt', 'gt']:  # 遍历每个不等号
                thresh_val = (range_min + float(j) * step_size)  # 取阈值
                predict_val = stump_classify(data_mat, i, thresh_val, inequal)  # 利用阈值进行分类
                err_arr = mat(ones((m, 1)))  # 初始错误率
                err_arr[predict_val == label_mat] = 0  # 预测正确将其归零
                weighted_error = D.T * err_arr  # 权值误差
                # print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f'
                #       % (i, thresh_val, inequal, weighted_error)
                #       )
                if weighted_error < min_err:  # 找到最小错误率
                    min_err = weighted_error
                    best_clas_est = predict_val.copy()  # 最佳分类
                    # 最佳单层决策树
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
                    print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f'
                          % (i, thresh_val, inequal, weighted_error)
                          )
                    print('-------------------------')
    return best_stump, min_err, best_clas_est


def adaboost_train(data_arr, class_labels, num_it=40):
    """
    基于单层决策树的训练过程
    :param data_arr:
    :param class_labels:
    :param num_it:
    :return:
    """
    weak_class_arr = []
    m = shape(data_arr)[0]
    D = mat(ones((m, 1)) / m)
    agg_class_est = mat(zeros((m, 1)))
    for i in range(num_it):
        best_stump, error, class_est = bulid_stump(data_arr, class_labels, D)
        print('D:', D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)  # 最佳单层决策树保存
        print('class_est:', class_est.T)
        expon = multiply(-1 * alpha * mat(class_labels).T, class_est)
        D = multiply(D, exp(expon))
        D = D / D.sum()  # 更新训练集权值
        agg_class_est += alpha * class_est  # 基本分类器的线性组合
        print('aggClassEst:', agg_class_est.T)
        agg_errors = multiply(sign(agg_class_est) != mat(class_labels).T, ones((m, 1)))
        error_rate = agg_errors.sum() / m
        print('total error:', error_rate, '\n')
        if error_rate == 0.0:
            break
    return weak_class_arr, agg_class_est


def ada_classifier(data2class, classifier_arr):
    """
    adaboost分类函数
    :param data2class:
    :param classifier_arr:
    :return:
    """
    data_mat = mat(data2class)
    m = shape(data_mat)[0]
    agg_class_est = mat(zeros((m, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(data_mat, classifier_arr[i]['dim'], classifier_arr[i]['thresh'],
                                   classifier_arr[i]['ineq'])  # 利用训练好的模型，预测最佳分类
        agg_class_est += classifier_arr[i]['alpha'] * class_est  # 利用线性组合计算最终聚合分类
    print('--------------')
    print(agg_class_est)
    print(sign(agg_class_est))
    print(shape(agg_class_est)[0])
    return sign(agg_class_est)


def load_dataset(file_name):
    num_feat = len(open(file_name).readline().split('\t'))
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat - 1):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def plot_ROC(pred_strength, class_labels):
    """
    :param pred_strength: 预测强度
    :param class_labels: 标签
    :return:
    """
    cur = (1.0, 1.0)
    y_sum = 0.0
    num_pos_clas = sum(array(class_labels) == 1.0)
    y_step = 1 / float(num_pos_clas)
    x_step = 1 / float(len(class_labels) - num_pos_clas)
    sorted_indices = pred_strength.argsort()  # 获取排好序的索引
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_indices.tolist()[0]:
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0]-del_x], [cur[1], cur[1]-del_y], c='b')
        cur = (cur[0] - del_x, cur[1]-del_y)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", y_sum * x_step)


def plot_red_circle():
    turtle.speed(0)
    turtle.hideturtle()
    turtle.bgcolor('black')
    i = 1
    color = ['red', 'purple', 'blue', 'white', 'green', 'pink', 'yellow', 'orange']
    while i:
        if i > 150:
            turtle.pencolor('orange')
        else:
            turtle.pencolor(random.randint(0, 1), random.randint(0, 1), random.randint(0, 1))

        turtle.penup()
        turtle.goto(0, 0)
        turtle.forward(200)
        turtle.pendown()
        turtle.circle(100)
        turtle.left(2)
        i += 1


if __name__ == '__main__':
    # D = mat(ones((5, 1))/5)
    data_mat, label = load_dataset('horseColicTraining2.txt')
    # print(bulid_stump(data_mat, label, D))
    classifier, agg_class_est = adaboost_train(data_mat, label, 10)
    # plot_ROC(agg_class_est.T, label)
    # print(classifier)
    test_mat, test_label = load_dataset('horseColicTest2.txt')
    prediton = ada_classifier(test_mat, classifier)
    err_arr = mat(ones((67, 1)))
    error = err_arr[prediton != mat(test_label).T].sum()
    print(error/67)
    # plot_red_circle()
