import matplotlib.pyplot as plt
import numpy as np


def t1():
    random = np.random.random(100)
    y = np.linspace(1, len(random), len(random))
    plt.plot(y, random)
    plt.show()


def t2():
    randn = np.random.randn(100)
    random = np.random.random(100)
    median = get_median(random)
    print(median)


def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:  # 判断列表长度为偶数
        median = (data[size // 2] + data[size // 2 - 1]) / 2
        data[0] = median
    if size % 2 == 1:  # 判断列表长度为奇数
        median = data[(size - 1) // 2]
        data[0] = median
    return data[0]


def t3():
    arange = np.arange(0.5, 1, 0.1)
    print(arange)


def t4():
    t = [1, 2, 3, 4, 2, 3, 1, 5]
    t.sort()
    print(t)

if __name__ == '__main__':
    t4()
