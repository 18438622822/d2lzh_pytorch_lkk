# -*- coding:utf-8 –*-
'''
@Author: lkk
@Date: 2019-12-18 15:39:51
@LastEditTime : 2019-12-18 15:39:54
@LastEditors  : lkk
@Description: 
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import d2lzh_pytorch as d2l


def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))


def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition


def relu(x):
    return torch.max(input=x, other=torch.tensor(0.0))
    # return x.relu()


def net(x):
    h = relu(torch.mm(x.view((-1, num_inputs)), w1) + b1)
    # h = torch.tensor(h, dtype=torch.float, requires_grad=True)
    return softmax(torch.mm(h, w2) + b2)


num_inputs, num_outputs, num_hiddens = 784, 10, 256
w1 = torch.normal(0, 0.01, (num_inputs, num_hiddens), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
w2 = torch.normal(0, 0.01, (num_hiddens, num_outputs), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)
params = [w1, b1, w2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)
loss = cross_entropy
num_epochs, lr = 5, 0.5
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params,
              lr)
