#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: Ubuntu18.04
@编程环境：python3.7
@Created on 2018-12-10 20:32
@Author:ChileWang
@algorithm:Apriori
原理：一个项集是非频繁的，那他的超集一定是非频繁的。
亦即是知道非频繁项集的支持度，其超集的支持度无需再求，定是不满足条件的。
支持度 = 包含该项集的集合个数 / 集合总个数

伪代码：
    对数据集的每条记录tran
        对每个候选集can：
            检查can是否属于tran的子集
            如果是，增加can的计数值
    对每个候选集：
        如果其支持度大于最小支持度，则保留该项集
    返回所有频繁项集列表

关联规则也同理，（0，1，2）->3小于我们所期望的可信度，那么其子集也小于我们所期望的可信度。
P->H的可信度 = support(P | H) / support(P)  '|' 是集合的并操作
"""


def load_data_set():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(data_set):
    """
    产生大小为1的不重复候选项
    :param data_set:
    :return:
    """
    c1 = []
    for transaction in data_set:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    return list(map(frozenset, c1))  # 对C1每个项构建一个不可变（冻结）的项集


def scan_d(d, ck, min_support):
    """
    :param d:数据集
    :param ck:候选项集
    :param min_support:最小支持度
    :return:返回符合最小支持度的候选项集及其支持度
    """
    ss_cnt = {}  # 存放候选项集合在数据集合的出现次数
    for tid in d:
        for can in ck:
            if can.issubset(tid):  # 计算候选项集在数据集中的次数
                if can not in ss_cnt.keys():
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1
    num_items = float(len(d))
    ret_list = []
    support_data = {}

    for key in ss_cnt:
        support = ss_cnt[key] / num_items  # 项集支持度
        if support >= min_support:
            ret_list.insert(0, key)  # 首部插入
            support_data[key] = support
    return ret_list, support_data


def apriori_gen(lk, k):
    """
    合并候选集项，产生新的候选集
    :param lk: 频繁项集列表
    :param k:项集元素个数
    :return:
    """
    ret_list = []
    len_lk = len(lk)
    for i in range(len_lk):
        for j in range(i + 1, len_lk):  # 比较lk中每一个元素与其他元素
            l1 = list(lk[i])[: k - 2]
            l2 = list(lk[j])[: k - 2]
            l1.sort()
            l2.sort()
            if l1 == l2:  # 前 k-2个项相等，便可合并，目的是无需便利整个列表来寻找非重复值
                ret_list.append(lk[i] | lk[j])
    return ret_list


def apriori(data_set, min_support=0.5):
    """
    :param data_set:
    :param min_support:
    :return:生成合并的候选项列表l 和对应的支持度
    """
    c1 = create_c1(data_set)
    d = list(map(set, data_set))  # 将列表的每一项都变成集合
    l1, support_data = scan_d(d, c1, min_support)
    l = [l1]  # 合并的候选项都放在l中， 后续还会有l2, l3.....
    k = 2  # 新构建的项集所含有的项目数
    while len(l[k - 2]):
        ck = apriori_gen(l[k - 2], k)  # 产生比原来项集的元素列表个数更大的候选集
        lk, sup_k = scan_d(d, ck, min_support)  # 删除不符合最小支持度的候选项集
        support_data.update(sup_k)  # 将字典sup_k的键/值对更新到dict里。
        l.append(lk)
        k += 1
    return l, support_data


def generate_rules(l, support_data, min_conf=0.7):
    """
    关联规则生成函数
    :param l: 频繁项集列表
    :param support_data: 每个项集（包括非频繁项集）的支持度
    :param min_conf: 期望的最低可信度
    :return:
    """
    big_rule_list = []
    for i in range(1, len(l)):  # 只获取项集中有两个或者更多元素的频繁项集
        for freq_set in l[i]:  # 频繁项集列表的元素
            H1 = [frozenset([item]) for item in freq_set]
            if i > 1:  # 如果频繁项集的元素超过2， 考虑将其合并
                rule_from_conseq(freq_set, H1, support_data, big_rule_list, min_conf)
            else:
                calc_conf(freq_set, H1, support_data, big_rule_list, min_conf)
    return big_rule_list


def calc_conf(freq_set, H1, support_data, big_rule_list, min_conf):
    """
    计算可信度
    :param freq_set:项集的一个元素
    :param H1:只包含单个元素的列表
    :param support_data:
    :param big_rule_list:
    :param min_conf:
    :return:
    """
    pruned_H = []  # 满足最小可信度要求的规则列表
    for conseq in H1:
        # P->H的可信度 = support(P | H) / support(P)  '|' 是集合的并操作
        conf = support_data[freq_set] / support_data[freq_set - conseq]  # '-' 并集中的交
        if conf > min_conf:
            print(freq_set - conseq, '-->', conseq, 'conf:', conf)
            big_rule_list.append((freq_set - conseq, conseq, conf))
            pruned_H.append(conseq)
    return pruned_H


def rule_from_conseq(freq_set, H, support_data, big_rule_list, min_conf):
    """
    :param freq_set: 
    :param H: 元素列表H
    :param support_data: 
    :param big_rule_list: 
    :param min_conf: 
    :return: 
    """
    m = len(H[0])  # 查看列表中的元素列表中的个数
    if len(freq_set) > m + 1:  # 查看频繁项集的元素个数是否大到可以删除大小为m的子集
        Hmp1 = apriori_gen(H, m + 1)  # 利用这个函数生成无重复的元素组合 # 包含所有可能生成的规则
        Hmp1 = calc_conf(freq_set, Hmp1, support_data, big_rule_list, min_conf)
        if len(Hmp1) > 1:
            rule_from_conseq(freq_set, Hmp1, support_data, big_rule_list, min_conf)


if __name__ == '__main__':
    data_set = load_data_set()
    c1 = create_c1(data_set)
    D = list(map(set, data_set))
    l, support_data = apriori(data_set, 0.1)
    print(l)
    print(support_data)
    rule = generate_rules(l, support_data, min_conf=0)
    print('------------------')
    mush_set = [line.split() for line in open('mushroom.dat').readlines()]
    l, support_data = apriori(mush_set, 0.3)





