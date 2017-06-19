# -*- coding: UTF-8 -*-
"""
Created on 2017-06-19

@author: zzx
"""

import numpy as np


class Node(object):
    def __init__(self, node_data, split, left=None, right=None):
        self.node_data = node_data
        self.split = split
        self.left = left
        self.right = right


class KdTree(object):
    def __init__(self, length):
        self.length = length

    # 递归建立KD树
    def create_tree(self, data):
        m = data.shape[0]
        if m == 0:
            return None

        split = np.argmax(np.var(data, axis=0))
        sorted_data = np.array(sorted(data, key=lambda x: x[split]))
        index = m/2

        return Node(sorted_data[index, :], split,
                    left=self.create_tree(sorted_data[0:index, :]),
                    right=self.create_tree(sorted_data[index+1:, :]))

    # 前序遍历
    def first_walk(self, root):
        if root:
            print root.node_data
        if root.left:
            self.first_walk(root.left)
        if root.right:
            self.first_walk(root.right)

    # 某个点的二分查找路径
    def search_path(self, root, x, path):
        current_node = root
        while current_node:
            path.append(current_node)
            split = current_node.split
            if x[split] < current_node.node_data[split]:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return path

    # 最近邻搜索
    def find_nn(self, root, x):
        path = self.search_path(root, x, [])
        nearest_node = path.pop()
        nearest_dist = self.distance(x, nearest_node.node_data)

        while path:
            current_node = path.pop()
            current_dist = self.distance(x, current_node.node_data)
            if current_dist < nearest_dist:
                nearest_node = current_node
                nearest_dist = current_dist

            split = current_node.split
            if abs(x[split]-current_node.node_data[split]) < nearest_dist:
                if x[split] < current_node.node_data[split]:
                    path = self.search_path(current_node.right, x, path)
                else:
                    path =self.search_path(current_node.left, x, path)

        return nearest_node.node_data


    # 欧式距离
    def distance(self, x, y):
        return np.sum(np.square(x-y))

def test():
    aa = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
    a = np.array(aa)
    t = KdTree(len(aa))
    root = t.create_tree(a)
    # t.first_walk(root)
    # t.search_path(root, np.array([7,2]), [])
    print t.find_nn(root, np.array([6.9, 2]))

if __name__ == "__main__":
    test()