# -*- coding:utf-8 –*-
'''
@Author: lkk
@Date: 2019-11-06 22:34:02
@LastEditTime: 2019-11-07 10:17:04
@LastEditors: lkk
@Description: 
'''
import random
from IPython import display
from matplotlib import pyplot as plt
import torch


def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i : min(i + batch_size, num_examples)]) # 类型转换为long和tensor
        yield features.index_select(0, j), labels.index_select(0, j) #index_select (dim, index_list/int), dim代表那个轴 输出与输入在轴的维度上相同