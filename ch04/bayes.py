#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: KALI LINUX
@编译器：python3.7
@Created on 2018-11-08 19:47
@Author:ChileWang
@algorithm：贝叶斯算法
"""
from numpy import *
import re
import os
import feedparser


def load_dataset():
    """
    数据集
    :return:
    """
    posting_list = [
                ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    class_vec = [0, 1, 0, 1, 0, 1]    # 1 is abusive, 0 not
    return posting_list, class_vec


def create_vocal_list(dataset):
    """
    :param dataset:
    :return:不重复词表
    """
    vocab_set = set([])
    for doc in dataset:
        vocab_set = vocab_set | set(doc)  # 并集
    return list(vocab_set)


def set_of_word2vec(vocal_list, input_set):
    """
    :param vocal_list:词汇表
    :param input_set:输入的测试文档,判断文档单词是否在词汇表中
    :return:文档词汇转换成文档向量
    """
    return_vec = [0] * len(vocal_list)  # 创建所含元素都为0的向量
    for word in input_set:
        if word in vocal_list:
            return_vec[vocal_list.index(word)] += 1
        else:
            print("the word : %s is not in my vocabulary!" % word)
    return return_vec


def train_nbc(train_mat, train_category):
    """
    p(c1/w) = p(w|c1)p(c1)/ p(w)  p(c1/w):该词汇表下，侮辱性文档的概率是多大
    :param train_mat: 训练文档
    :param train_category: 训练文档词条所对应的向量（1为侮辱，0为非侮辱）
    :return:
    """
    train_doc_num = len(train_mat)  # 训练文档数量
    train_word_num = len(train_mat[0])  # 训练文档的词量
    p_absive = sum(train_category) / float(train_doc_num)  # 侮辱性词汇的概率  p(c)
    p0_num = ones(train_word_num)
    p1_num = ones(train_word_num)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(train_doc_num):
        if train_category[i] == 1:  # p(w/c1）
            p1_num += train_mat[i]  # 统计侮辱性文档每个词语各自出现的次数
            p1_denom += sum(train_mat[i])  # 侮辱性文档出现的总的词量
        else:
            p0_num += train_mat[i]  # p(w/c2)
            p0_denom += sum(train_mat[i])

    p1_vect = log(p1_num / p1_denom)
    p0_vect = log(p0_num / p0_denom)

    return p0_vect, p1_vect, p_absive


def classify_nb(vec2classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec2classify * p1_vec) + log(p_class1)
    p0 = sum(vec2classify * p0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_nb():
    posting_list, list_class = load_dataset()
    my_vocab_list = create_vocal_list(posting_list)
    train_mat = []
    for post_in_doc in posting_list:
        train_mat.append(set_of_word2vec(my_vocab_list, post_in_doc))

    p0_v, p1_v, p_ab = train_nbc(array(train_mat), array(list_class))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_word2vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as :', classify_nb(this_doc, p0_v, p1_v, p_ab))
    test_entry = ['stupid', 'garbage']
    this_doc = array(set_of_word2vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as :', classify_nb(this_doc, p0_v, p1_v, p_ab))


def text_parse(big_str):
    """
    文本切分并转换为小写单词
    :param big_str:
    :return:
    """
    # spilt_str = ''
    # for sub_str in big_str:
    #     spilt_str += sub_str

    list_of_token = re.split(r'\W', big_str)
    return [tok.lower() for tok in list_of_token if len(tok) > 2]


def spam_text():
    doc_list = []
    class_list = []
    full_text = []
    spam_dir = 'email/spam/'
    ham_dir = 'email/ham/'
    open_spam_txt = os.listdir(spam_dir)
    open_ham_txt = os.listdir(ham_dir)

    # 导入并解析文本
    for txt in open_spam_txt:
        word_list = text_parse(open(spam_dir + txt, 'r', encoding='gbk').read())
        doc_list.append(word_list)  # append形成二维列表
        full_text.extend(word_list)  # extend形成一维列表
        class_list.append(1)

    for txt in open_ham_txt:
        word_list = text_parse(open(ham_dir + txt, 'r', encoding='gbk').read())
        doc_list.append(word_list)  # append形成二维列表
        full_text.extend(word_list)  # extend形成一维列表
        class_list.append(0)
    vocab_list = create_vocal_list(doc_list)
    # 随机构建训练集
    train_set = list(range(50))
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[rand_index])
        del(train_set[rand_index])

    train_mat = []
    train_classes = []
    for doc_index in train_set:
        train_mat.append(set_of_word2vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_nbc(array(train_mat), array(train_classes))

    # 统计错误率
    error_count = 0
    for doc_index in test_set:
        word_vec = set_of_word2vec(vocab_list, doc_list[doc_index])
        if classify_nb(array(word_vec), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    # print('the eror rate is ', (float(error_count) / len(test_set)))

    return float(error_count) / len(test_set)


def cal_most_freq(volcab_list, full_text):
    """
    统计高频词汇
    :param volcab_list:词汇表
    :param full_text:文章
    :return:
    """
    freq_dict = dict()
    for token in volcab_list:
        freq_dict[token] = full_text.count(token)
    sorted_fre = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)  # 字典按频率倒序排序
    return sorted_fre[:30]


def local_words(feed1, feed0):
    doc_list = []  # 文件表
    class_list = []  # 分类表
    full_text = []  # 全文
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        word_list = text_parse(feed1['entries'][i]['summary'])  # 每次访问一条RSS源
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)

        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    vocab_list = create_vocal_list(doc_list)  # 创建词汇表
    top30words = cal_most_freq(vocab_list, full_text)  # 统计所有词汇中最高词频率前30，删除前30保证常用词汇不影响结果
    for pairw in top30words:  # pairw[0]是键，即词
        print(pairw)
        if pairw[0] in vocab_list:
            vocab_list.remove(pairw[0])

    #  随机创建训练集
    train_set = list(range(2 * min_len))
    test_set = []
    for i in range(20):
        rand_index = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[rand_index])  # 创建测试集合
        del(train_set[rand_index])

    train_mat = []
    train_classfy = []

    for doc_index in train_set:
        train_mat.append(set_of_word2vec(vocab_list, doc_list[doc_index]))  # 创建训练集
        train_classfy.append(class_list[doc_index])

    p0_v, p1_v, p_spam = train_nbc(array(train_mat), array(train_classfy))

    # 统计错误率
    error_count = 0
    for doc_index in test_set:
        word_vec = set_of_word2vec(vocab_list, doc_list[doc_index])
        if classify_nb(array(word_vec), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    # print('the eror rate is ', (float(error_count) / len(test_set)))

    return float(error_count) / len(test_set)



if __name__ == '__main__':
    # aver_error_rate = 0.0
    # for i in range(20):
    #     aver_error_rate += spam_text()
    # print('the eror rate is ', aver_error_rate/20.0)


