# -*- coding:utf-8 –*-
'''
@Author: lkk
@Date: 2019-11-06 22:34:02
@LastEditTime: 2019-12-16 17:38:14
@LastEditors: lkk
@Description: 
'''
import sys
import random
from IPython import display
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    # use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(
            indices[i:min(i + batch_size, num_examples)])  # 类型转换为long和tensor
        yield features.index_select(0, j), labels.index_select(
            0, j)  # index_select (dim, index_list/int), dim代表那个轴 输出与输入在轴的维度上相同


def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def get_fashion_mnist_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def load_data_fashion_mnist(batch_size):
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./DataSets/FashionMNIST",
        train=True,
        download=True,
        transform=transforms.ToTensor())

    mnist_test = torchvision.datasets.FashionMNIST(
        root="./DataSets/FashionMNIST",
        train=False,
        download=True,
        transform=transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter
