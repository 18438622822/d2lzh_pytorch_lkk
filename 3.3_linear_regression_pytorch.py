import torch
import numpy as np
import torch.utils.data as Data
from torch import nn
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
# features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)))

features = torch.normal(0, 1, size=(num_examples, num_inputs), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
dev = torch.normal(0, 0.01, size=labels.size(), dtype=torch.float)
labels += dev
batch_size = 10
datasets = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(datasets, batch_size=batch_size, shuffle=True)


class LinearNet(nn.Module):
    def __init__(self, features):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(features, 1)

    def forward(self, x):
        y = self.linear(x)
        return y




net = nn.Sequential(nn.Linear(num_inputs, 1))

net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))

from collections import OrderedDict
net = nn.Sequential(OrderedDict([('linear', nn.Linear(num_inputs, 1))]))

net = LinearNet(num_inputs)
# print(net)

for param in net.parameters():
    print(param)

from torch.nn import init
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, 0)

## nn.Sequential
# init.normal_(net[0].weight, mean=0, std=0.01)
# init.constant_(net[0].bias, 0)

loss = nn.MSELoss()

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for x, y in data_iter:
        out = net(x)
        l = loss(out, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d loss: %f' % (epoch, l.item()))


print(true_w, net.linear.weight)
print(true_b, net.linear.bias)