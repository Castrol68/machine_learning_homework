#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: KALI LINUX
@编译器：python3.7
@Created on 2018-11-23 15:26
@Author:ChileWang
@algorithm：天池o2o优惠券预测赛
"""
import os
import pickle
from datetime import date
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split, \
    StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm


pd.set_option('display.max_columns', None)  # 显示所有列


def get_discount_tpye(row):
    """
    获取折扣类型
    :param row:
    :return:
    """
    if ':' in row:
        return 1
    elif row == 'null':
        return 'null'
    else:
        return 0


def conver_rate(row):
    """
    将折扣转化成折扣率，从而方便计算
    :param row:
    :return:
    """
    if row == 'null':
        return 1.0
    elif ':' in row:
        rows = row.split(':')
        return 1.0 - float(rows[1]) / float(rows[0])
    else:
        return float(row)


def get_discount_man(row):
    """
    获得折扣类型中的满减类型中的满
    :param row:
    :return:
    """
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    return 0


def get_discount_jian(row):
    """
    获得折扣类型中的满减类型中的减
    :param row:
    :return:
    """
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    return 0


def label(row):
    """
    给数据上标签
    :param row:
    :return:
    """
    if row['Date_received'] == 'null':  # 无优惠券的直接不考虑了，我们的目的是让优惠券去到需要的人手里
        return -1
    if row['Date'] != 'null':
        #  计算是否在优惠券的使用期限内使用
        td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.to_timedelta(15, 'D'):
            return 1
    return 0


def get_weekday(row):
    """
    获取具体的星期几
    :param df:
    :return:
    """
    if row == 'null':
        return row
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1


def process_data(df):
    """
    将数据集转化成统一数值型特征
    :param df:
    :return:
    """
    df['discount_type'] = df['Discount_rate'].apply(get_discount_tpye)  # 折扣类型
    df['discount_rate'] = df['Discount_rate'].apply(conver_rate)  # 折扣率
    df['discount_man'] = df['Discount_rate'].apply(get_discount_man)  # 折扣率中的满
    df['discount_jian'] = df['Discount_rate'].apply(get_discount_jian)  # 折扣率中的减
    df['distance'] = df['Distance'].replace('null', -1).astype(int)  # 距离
    df['weekday'] = df['Date_received'].astype(str).apply(get_weekday)  # 转换成具体星期几
    df['weekday_type'] = df['weekday'].astype(str).apply(lambda x: 1 if x in [6, 7] else 0)  # 周末为1，工作日为0
    # 将星期几进行独热编码，相当于归一化
    weekday_cols = ['weekday_' + str(i) for i in range(1, 8)]
    tmpdf = pd.get_dummies(df['weekday'].replace('null', np.nan))
    tmpdf.columns = weekday_cols
    df[weekday_cols] = tmpdf

    original_feature = ['discount_rate', 'discount_type', 'discount_man', 'discount_jian', 'distance', 'weekday',
                        'weekday_type'] + weekday_cols
    print('共有特征：', len(original_feature), '个')
    print(original_feature)

    return df, original_feature


def data_split(df):
    """
    将训练集分为训练集和验证集
    :return:
    """
    df = df[df['label'] != -1].copy()  # 将没有优惠券的人群剔除
    train_set = df[df['Date_received'] < '20160516'].copy()
    valid_set = df[(df['Date_received'] >= '20160516') & (df['Date_received'] <= '20160615')].copy()
    return train_set, valid_set


def check_model(data, preditors):
    """
    模型预测
    :param data: 数据集合
    :param preditors: 特征
    :return:拟合的SDG模型
    """
    classifier = lambda: SGDClassifier(
        loss='log',  # loss funtion:logistic regression
        penalty='elasticnet',  # L1 & L2
        fit_intercept=True,  # 是否存在截距，默认存在
        max_iter=100,
        shuffle=True,  # Whether or not the training data should be shuffled after each epoch
        n_jobs=1,  # The number of processors to use
        class_weight=None  # Weights associated with classes. If not given, all classes are supposed to have weight one.
    )

    # 管道机制使得参数集在新数据集（比如测试集）上的重复使用，管道机制实现了对全部步骤的流式化封装和管理。
    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        # transformer  # SGDClassifier对于特征的幅度非常敏感，也就是说，
        # 我们在把数据灌给它之前，应该先对特征做幅度调整，
        # 当然，用sklearn的StandardScaler可以很方便地完成这一点
        ('en', classifier())  # estimator
    ])

    parameters = {
        'en__alpha': [0.001, 0.01, 0.1],
        'en__l1_ratio': [0.001, 0.01, 0.1]
    }
    # StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
    folder = StratifiedKFold(n_splits=3, shuffle=True)
    # Exhaustive search over specified parameter values for an estimator.
    grid_search = GridSearchCV(
        model,  # estimator：使用的模型
        parameters,
        cv=folder,  # cv：交叉验证参数，默认为3，使用3折交叉验证
        n_jobs=-1,  # -1 means using all processors
        verbose=1)  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。
    grid_search = grid_search.fit(data[preditors],
                                  data['label'])
    return grid_search


def checke_svm_model(dataset, preditors):
    """
    模型预测
    :param dataset: 数据集合
    :param preditors: 特征
    :return:拟合的SVM模型
    """

    parameters = [{'en__C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],  # 惩罚因子
                   'en__gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
                   # rbf自带参数，与C一样，越大支持向量越多，使得高斯分布又高又瘦，造成模型只能作用于支持向量附近，也就是容易过拟合
                   'en__kernel': ['rbf']},
                  {'en__C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 'en__kernel': ['linear']}]  # 要改成en__+变量，规定的。

    svr = svm.SVC()
    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        # transformer  # SGDClassifier对于特征的幅度非常敏感，也就是说，
        # 我们在把数据灌给它之前，应该先对特征做幅度调整，
        # 当然，用sklearn的StandardScaler可以很方便地完成这一点
        ('en', svr)  # estimator
    ])
    clf = GridSearchCV(model, parameters, n_jobs=-1, verbose=1)
    clf.fit(dataset[preditors], dataset['label'])
    best_model = clf.best_estimator_
    """
    clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8) 
    clf.fit(train_data, train_data_tag) 
    print(clf.best_params_) 
    best_model = clf.best_estimator_ 
    best_model.predict(test_data)
    """
    return best_model


def save_model(model):
    """
    保存模型
    :param model:
    :return:
    """
    if not os.path.isfile('modle_1.pkl'):
        with open('modle_1.pkl', 'wb') as f:
            pickle.dump(model, f)


def get_model():
    """
    获得模型
    :return:
    """
    with open('modle_1.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def main():
    dfoff = pd.read_csv('ccf_offline_stage1_train.csv', keep_default_na=False)  # 训练集
    dftest = pd.read_csv('ccf_offline_stage1_test_revised.csv', keep_default_na=False)  # 测试集
    print(dfoff.head(5))
    print('有优惠券，购买商品:%d' % dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] != 'null')].shape[0])
    print('有优惠券，不买商品:%d' % dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] == 'null')].shape[0])
    print('无优惠券，购买商品:%d' % dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] != 'null')].shape[0])
    print('无优惠券，不买商品:%d' % dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] == 'null')].shape[0])

    print('Discount rate 类型\n', dfoff['Discount_rate'].unique())  # 折扣类型
    print('Distance:\n', dfoff['Distance'].unique())  # 距离类型
    print('Date_received:\n', dfoff['Date_received'].unique())  # 类型

    date_received = dfoff['Date_received'].unique()
    date_received = sorted(date_received[date_received != 'null'])

    date_buy = dfoff['Date'].unique()
    date_buy = sorted(date_buy[date_buy != 'null'])

    print('优惠卷收到日期从', date_received[0], '到', date_received[-1])
    print('消费日期从', date_buy[0], '到', date_buy[-1])

    dfoff['label'] = dfoff.apply(label, axis=1)  # 给训练集上标签
    # print(dfoff['label'].value_counts())

    df_off, original_feature = process_data(dfoff)  # 将训练集的特征转化成统一的数值特征
    df_test, original_feature = process_data(dftest)  # 将测试集的特征转化成统一的数值特征

    train_set, valid_set = data_split(df_off)  # 将训练分为训练集和验证集
    print('Train Set: \n', train_set['label'].value_counts())
    print('Valid Set: \n', valid_set['label'].value_counts())

    predictors = original_feature  # 特征

    # 训练
    # model = check_model(train_set, predictors)
    # model = get_model()
    model = checke_svm_model(train_set, predictors)
    print('Train is done!')
    # 验证
    y_valid_pred = model.predict_proba(valid_set[predictors])
    # predict_proba返回的是一个 n 行 k 列的数组，
    # 第 i 行 第 j 列上的数值是模型预测 第 i 个预测样本为某个标签的概率，并且每一行的概率和为1。
    y_label = model.predict(valid_set[predictors])
    print(y_valid_pred)
    print(y_label)
    valid_set_1 = valid_set.copy()
    valid_set_1['pred_prob'] = y_valid_pred[:, 1]
    # valid_set_1['y_label'] = y_label[:, 1]
    # print(valid_set_1['y_label'].unique())
    # print(valid_set_1[valid_set_1['pred_prob'] <= 0.5].shape[0])
    # 计算AUC
    vg = valid_set_1.groupby(['Coupon_id'])  # 0.53
    aucs = []
    for i in vg:
        tmpdf = i[1]
        if len(tmpdf['label'].unique()) != 2:
            continue
        fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
        aucs.append(auc(fpr, tpr))
    print(np.average(aucs))

    # 测试
    y_test_pred = model.predict_proba(df_test[predictors])
    dftest1 = df_test[['User_id', 'Coupon_id', 'Date_received']].copy()
    dftest1['Probability'] = y_test_pred[:, 1]
    dftest1.to_csv('submit1.csv', index=False, header=False)
    print(dftest1[dftest1['Probability'] > 0.5].shape[0])
    print(dftest1[dftest1['Probability'] <= 0.5].shape[0])
    # 保存模型
    save_model(model)


if __name__ == '__main__':
    main()
