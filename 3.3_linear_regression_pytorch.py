import torch
import numpy as np
import torch.utils.data as Data
num_inputs = 2
num_examples = 10
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
        super()