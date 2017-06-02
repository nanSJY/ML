# -*- coding: UTF-8 -*-

'''
Created on 2017年2月9日

@author: nanSJY
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def creat_hypothesis(x, theta):
    sum = 0
    for i in range(len(x)):
        sum += x[i] * theta[i]
    return sum


def linear_regression(data, alpha, epsilon):
    m = len(data)  # 样本数量
    n = len(data[0]) - 2  # 特征个数
    x = [data[i][0:-1] for i in range(m)]
    y = [data[i][-1] for i in range(m)]

    theta = [0. for i in range(n + 1)]
    error = []  # 每次迭代的误差
    count = 0  # 迭代次数

    while (1):
        count += 1

        # 计算theta
        error_term = []
        for i in range(m):
            error_term.append(y[i] - creat_hypothesis(x[i], theta))
        error.append(sum([i * i for i in error_term]))

        for j in range(n + 1):
            diff = 0
            for i in range(m):
                diff = diff + error_term[i] * x[i][j]
            theta[j] = theta[j] + diff * alpha

        # 判断是否收敛
        if (len(error) <= 1):
            continue
        elif abs(error[-1] - error[-2]) < epsilon:
            break
    print 'theta =', theta
    print '迭代次数  =', count
    draw(x, y, theta, count, error, m)


def draw(x, y, theta, count, error, m):
    # cost function
    plt.figure(1)
    num = [i for i in range(count)]
    plt.plot(num, error, linewidth='3')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.title("cost function")
    if len(theta) == 3:
        # n = 2
        # 样本点
        fig2 = plt.figure()
        ax2 = Axes3D(fig2)
        dx1 = [data[i][1] for i in range(m)]
        dx2 = [data[i][2] for i in range(m)]
        dy = y
        ax2.scatter(dx1, dx2, dy, s=100, c='#EE2C2C')
        # 模型
        x = [i for i in range(400)]
        y = x
        x, y = np.meshgrid(x, y)
        h = theta[0] + theta[1] * x + theta[2] * y
        ax2.plot_surface(x, y, h, color='#C0FF3E')

    if len(theta) == 2:
        # n = 1
        plt.figure(2)
        ax = plt.subplot(111)
        # 样本点
        dx = [data[i][1] for i in range(m)]
        dy = y
        ax.scatter(dx, dy, label="training set")
        # 模型
        num = [i / 100. for i in range(2500)]
        h = [theta[0] + theta[1] * i for i in num]
        ax.plot(num, h, c='r', label='h(x)')

    plt.legend()
    plt.show()


def loadData(dir):
    data = []
    file = open(dir)
    for line in file.readlines():
        x, y = line.strip().split(',')
        data.append([1., float(x), float(y)])
    return data


if __name__ == '__main__':
    from sklearn import datasets
    boston = datasets.load_boston()

    # data = loadData("E:\Python\workspace\MachineLearning\LinearRegression\data.txt")
    linear_regression(boston, alpha=0.0001, epsilon=0.0000000000000000000000001)