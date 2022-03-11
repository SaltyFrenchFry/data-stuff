# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:46:27 2022

@author: lixin
"""

import torch
from torch.nn import Linear
import torch.nn as nn

w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(-1.0, requires_grad = True)

def forward(x):
    y = w*x+b
    return y

'''
x = torch.tensor([1.0])
yhat = forward(x)
print(yhat)


x = torch.tensor([[1],[2]])
yhat = forward(x)
print(yhat)
'''

torch.manual_seed(1)

model = Linear(in_features = 1, out_features = 1)
#print(list(model.parameters()))

x = torch.tensor([0.0])
z = torch.tensor([[1.0],[2.0]])

yhat = model(z)
#print(yhat)


class LR(nn.Module):
    def __init__(self, in_size, out_size):
        super(LR, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out

model = LR(1, 1)
model.state_dict()['linear.weight'].data[0] = torch.tensor([0.5153])
model.state_dict()['linear.bias'].data[0] = torch.tensor([-0.4414])

f = torch.tensor([1.0])
yhat = model(f)
#print(yhat)

print("Python dictionary:", model.state_dict())
print("keys:", model.state_dict().keys())
print("values:",model.state_dict().values())