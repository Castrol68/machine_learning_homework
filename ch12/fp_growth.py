#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: Ubuntu18.04
@编译环境：python3.7
@Created on 2018-11-23 15:26
@Author:ChileWang
@algorithm：fp_growth
"""


class tree_node:
    def __init__(self, name_value, num_occur, parent_node):
        """
        FP树类的定义
        :param name_value: 节点名字
        :param num_occur: 出现次数
        :param parent_node: 父节点
        """
        self.name = name_value
        self.count = num_occur
        self.node_link = None  # 链接相似的元素项
        self.parent = parent_node
        self.children = {}

    def inc(self, num_occur):
        self.count += num_occur

    def disp(self, ind=1):
        """
        显示子节点
        :param ind:
        :return:
        """
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def create_tree(dataset, min_s_up=1):
    """
    过滤--->排序--->建树
    :param dataset:
    :param min_s_up:最小支持度
    :return:
    """
    header_table = {}  # 扫描头指针表
    for trans in dataset:  # 每个事物
        for item in trans:  # 每个元素
            header_table[item] = header_table.get(item, 0) + dataset[trans]  # 统计每个元素出现的频率

    header_table_copy = header_table.copy()
    for k in header_table_copy.keys():  # 删除不满最小支持度的元素项目
        if header_table[k] < min_s_up:
            del(header_table[k])
    freq_item_set = set(header_table.keys())
    if len(freq_item_set) == 0:  # 如果没有元素项满足要求，则退出
        return None, None
    for k in header_table:
        header_table[k] = [header_table[k], None]  # 扩展为列表以便保存计数值以及指向每种类型的第一个元素项的指针
    ret_tree = tree_node('Null Set', 1, None)  # 创建只包含空集合的根节点

    for trans_set, count in dataset.items():
        local_id = {}
        for item in trans_set:  # 根据全局频率对每一个事物中的 元素 进行排序
            if item in freq_item_set:
                local_id[item] = header_table[item][0]  # 计数值
        if len(local_id) > 0:
            ordered_items = [v[0] for v in sorted(local_id.items(), key=lambda p:p[1], reverse=True)]  # 按值域倒序排序
            update_tree(ordered_items, ret_tree, header_table, count)  # 使用排序后的频率项集进行填充
    return ret_tree, header_table


def update_tree(items, intree, header_table, count):
    if items[0] in intree.children:  # 存在则更新计数值
        intree.children[items[0]].inc(count)
    else:
        intree.children[items[0]] = tree_node(items[0], count, intree)
        if header_table[items[0]][1] == None:  # 更新头指针表指向
            header_table[items[0]][1] = intree.children[items[0]]
        else:
            update_header(header_table[items[0]][1], intree.children[items[0]])
    if len(items) > 1:
        update_tree(items[1::], intree.children[items[0]], header_table, count)  # 对剩余的元素项目迭代调用


def update_header(node_to_test, target_node):
    """
    更新头指针表指向
    :param node_to_test:
    :param target_node:
    :return:
    """
    while node_to_test.node_link:
        node_to_test = node_to_test.node_link
    node_to_test.node_link = target_node


def load_simp_dat():
    simp_dat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
                ]
    return simp_dat


def create_init_set(dataset):
    ret_dict = {}
    for trans in dataset:
        ret_dict[frozenset(trans)] = 1
    return ret_dict


def ascend_tree(leaf_node, prefix_path):
    """
    上溯整棵树
    :param leaf_node: 叶子节点
    :param prefix_path: 前缀路径
    :return:
    """
    if leaf_node.parent:
        prefix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, prefix_path)


def find_prefix_path(base_pat, tree_node):
    """
    :param base_pat:寻找的目标
    :param tree_node:treeNode comes from header table
    :return:
    """
    cond_pats = {}
    while tree_node:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            cond_pats[frozenset(prefix_path[1:])] = tree_node.count  # 前缀路径以及base_pat分配到该路径的计数值
        tree_node = tree_node.node_link  # header_table指向下一个
    return cond_pats


def mine_tree(in_tree, header_table, min_s_up, prefix, freq_item_list):
    """
    递归查找频繁项集
    :param in_tree: FP树
    :param header_table:头指针表
    :param min_s_up:最小支持度
    :param prefix:前缀路径
    :param freq_item_list:频繁项集列表
    :return:
    """
    big_l = [v[0] for v in sorted(header_table.items(), key=lambda p:p[0])]  # 顺序按值排序
    for base_pat in big_l:  # 对于每一个元素,从头指针的底端开始
        new_freq_set = prefix.copy()
        new_freq_set.add(base_pat)
        freq_item_list.append(new_freq_set)
        cond_patt_bases = find_prefix_path(base_pat, header_table[base_pat][1])
        my_cond_tree, my_head = create_tree(cond_patt_bases, min_s_up)  # 利用剩下的元素重建FP树
        if my_head:
            print('conditional tree for ', new_freq_set)
            mine_tree(my_cond_tree, my_head, min_s_up, new_freq_set, freq_item_list)  # 头指针表不为空，则继续挖掘频繁项


if __name__ == '__main__':
    root_node = tree_node('pyramid', 9, None)
    root_node.children['eye'] = tree_node('eye', 13, None)
    root_node.disp()
    a = {'a': 1, 'b': 2, 'c': 3}
    a = [v[0] for v in sorted(a.items(), key=lambda p: p[1], reverse=True)]
    print(a[0])
    simple_data = load_simp_dat()
    init_set = create_init_set(simple_data)
    print(init_set)
    my_tree, my_header_table = create_tree(init_set, 3)
    my_tree.disp()
    print(my_header_table)
    freq_items = []
    mine_tree(my_tree, my_header_table, 3, set([]), freq_items)
    print(freq_items)
    parsed_dat = [line.split() for line in open('kosarak.dat').readlines()]
    init_set = create_init_set(parsed_dat)
    my_freq_list = []
    my_fp_tree, my_header_table = create_tree(init_set, 100000)
    mine_tree(my_fp_tree, my_header_table, 100000, set([]), my_freq_list)
    print(len(my_freq_list))
    print(my_freq_list)
    print('done')