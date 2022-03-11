# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 09:09:53 2022

@author: lixin
"""

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out


criterion = nn.MSELoss()

class Data(Dataset):
    def __init__(self, train = True):
        self.X = torch.arange(-3, 3, 0.1).view(-1,1)
        self.Y = 3*self.X+1 + 0.3*torch.randn(self.X.size())
        self.len = self.X.shape[0]
    #y = 3x + 1
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len
    
dataset = Data()
#validationData = Data(train = False)

trainloader = DataLoader(dataset=dataset, batch_size=10)

w = torch.tensor(-60.0, requires_grad = True)
#b = torch.tensor(-100.0, requires_grad = True)

model = LR(1, 1)

optimizer = optim.SGD(model.parameters(), lr = 0.01)

torch.save(model.state_dict(), "best_model.pt")

min_loss = 100

epochs = 100
learningRates = [0.0001, 0.001, 0.01, 0.1, 1]

validationError = torch.zeros(len(learningRates))

testError = torch.zeros(len(learningRates))

MODELS = []

COST = []
for epoch in range(epochs):
    
    total = 0
    
    for x,y in trainloader:
        yhat = model(x)
        loss = criterion(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total += loss.item()
    
    if(total < min_loss):
        value = epoch
        min_loss = total
        torch.save(model.state_dict(),"best_model.pt")
    
    n = True
    for param in model.parameters():
        if(n):
            w = param.data
            n = False
        else:
            b = param.data
        
    plt.plot(dataset.X.numpy(), dataset.X.numpy()*w.item()+b.item())
    plt.plot(dataset.X.numpy(), dataset.Y.numpy(), "ro")
    plt.show()
    
    COST.append(total)

model_best = LR(1, 1)
model_best.load_state_dict(torch.load("best_model.pt"))

n = True
for param in model.parameters():
    if(n):
        w = param.data
        n = False
    else:
        b = param.data
    
plt.plot(dataset.X.numpy(), dataset.X.numpy()*w.item()+b.item())
plt.plot(dataset.X.numpy(), dataset.Y.numpy(), "ro")
plt.show()

print(COST)