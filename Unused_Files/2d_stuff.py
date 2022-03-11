# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 21:02:24 2022

@author: lixin
"""
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out

class Data2D(Dataset):
    def __init__(self):
        self.x = torch.zeros(60, 2)
        self.x[:,0] = torch.arange(-3, 3, 0.1)
        self.x[:,1] = torch.arange(-3, 3, 0.1)
        self.w = torch.tensor([[2.0],[3.0]])
        self.b = 1
        self.f = torch.mm(self.x, self.w)+self.b
        self.y = self.f + 0.1*torch.randn((self.x.shape[0],1))
        self.len = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len

dataset = Data2D()

criterion = nn.MSELoss()

trainloader = DataLoader(dataset = dataset, batch_size = 2)

model = LR(input_size = 2, output_size = 1)

optimizer = optim.SGD(model.parameters(), lr = 0.01)

min_loss = 1000

COST = []
for epoch in range(1000):
    
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
    
        
    COST.append(total)

model_best = LR(2, 1)
model_best.load_state_dict(torch.load("best_model.pt"))

n = 0
for param in model_best.parameters():
    if(n == 0):
        w = param.data
        n = 1
    else:
        b = param.data

print(w)
print(b)
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i in range(len(dataset)):
    ax.scatter(dataset.x[i,0], dataset.x[i,1], dataset.y[i])

    
xx, yy = np.meshgrid(np.arange(-3,3,0.1), np.arange(-3,3,0.1))
z = xx*(w[0][0].item()) + yy*(w[0][1].item()) + b.item()

ax.plot_surface(xx, yy, z)

plt.show()

# yhat = w1x1 + w2x2 +... wnxn + b