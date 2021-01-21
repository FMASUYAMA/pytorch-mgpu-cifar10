# -*- coding: utf-8 -*-

'''simple model in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'basic':[120, 5, 16]
}
# config.hidden_nodes = 120
# config.conv1_channels = 5
# config.conv2_channels = 16

class Basic(nn.Module):
    def __init__(self, basic_name):
        super(Basic, self).__init__()
        self.conv1 = nn.Conv2d(3, cfg[basic_name][1], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, cfg[basic_name][2], 5)
        self.fc1 = nn.Linear(cfg[basic_name][2] * 5 * 5, cfg[basic_name][0])
        self.fc2 = nn.Linear(cfg[basic_name][0], 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test():
    print(cfg['basic'][0])
    net = Basic('basic')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

#test()