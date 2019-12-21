# -*- coding:utf-8 –*-
'''
@Author: lkk
@Date: 2019-12-18 15:04:31
@LastEditTime : 2019-12-18 15:04:34
@LastEditors  : lkk
@Description: 
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import d2lzh_pytorch as d2l


def xyplot(x, y, name):
    d2l.set_figsize((5, 2.5))
    plt.plot(x.detach().numpy(), y.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    plt.show()


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')
y.sum().backward()
xyplot(x, x.grad, "grad of relu")

y = x.sigmoid()
xyplot(x, y, 'sigmoid')
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, "grad of sigmoid")

y = x.tanh()
xyplot(x, y, 'tanh')
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, "grad of tanh")