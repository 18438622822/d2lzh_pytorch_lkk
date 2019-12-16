# -*- coding:utf-8 –*-
"""
@Author: lkk
@Date: 2019-12-14 13:26:12
@LastEditTime: 2019-12-16 14:54:54
@LastEditors: lkk
@Description: 
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import d2lzh_pytorch as d2l
from PIL import Image

mnist_train = torchvision.datasets.FashionMNIST(
    root="./DataSets/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.ToTensor())

mnist_test = torchvision.datasets.FashionMNIST(
    root="./DataSets/FashionMNIST",
    train=False,
    download=True,
    transform=transforms.ToTensor()
    )


X, y = [], []
for i in range(10):
    X.append(mnist_test[i][0])
    y.append(mnist_test[i][1])
d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))

train_iter, test_iter = d2l.load_data_fashion_mnist(256)
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))