# -*- coding:utf-8 –*-
'''
@Author: lkk
@Date: 2019-11-05 22:44:02
@LastEditTime: 2019-11-07 10:32:07
@LastEditors: lkk
@Description: 
'''

import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
from d2lzh_pytorch import *
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
# print(features[0], labels[0])
# # set_figsize()
# plt.figure()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.pause(10)
lr = 0.03
num_epochs = 3
batch_size = 10
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
w.requires_grad_(True)
b = torch.zeros(1, dtype=torch.float32)
b.requires_grad_(True)


def linreg_net(w, x, b):
    return torch.mm(x, w) + b


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2
    

loss = squared_loss
net = linreg_net
for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(w, x, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(w, features, b), labels)
    print("epoch%d, loss%f" % (epoch + 1, train_l.mean().item()))
    
print(w, b)
    