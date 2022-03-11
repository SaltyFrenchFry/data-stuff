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
        self.x = torch.zeros(60, 60, 2)
        coor = torch.arange(-3, 3, 0.1)
        for i in range(coor.shape[0]):
            for j in range(coor.shape[0]):
                self.x[i,j,0] = coor[i].item()
                self.x[i,j,1] = coor[j].item()
        self.w = torch.tensor([[3.0],[2.0]])
        self.b = 0
        self.f = torch.zeros(60, 60)
        for i in range(self.f.shape[0]):
            for j in range(self.f.shape[0]):
                self.f[i,j] = torch.dot(self.x[i,j], self.w.view(1,-1)[0])+self.b
        self.y = self.f + 0.5*torch.randn((self.x.shape[0],1))
        self.len = self.x.shape[0]
        
        print(self.x.dtype)
        print(self.y.dtype)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len

dataset = Data2D()

criterion = nn.MSELoss()

trainloader = DataLoader(dataset = dataset, batch_size = 5)

model = LR(input_size = 2, output_size = 1)

optimizer = optim.SGD(model.parameters(), lr = 0.01)

min_loss = 100000

COST = []
for epoch in range(10):
    
    total = 0
    
    for x,y in trainloader:
        yhat = model(x)
        loss = criterion(yhat.view(trainloader.batch_size,dataset.len), y)
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
print(COST)
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


x1 = dataset.x[:,:,0].view(-1,1).numpy()
x2 = dataset.x[:,:,1].view(-1,1).numpy()
y = dataset.y.view(-1,1).numpy()

for i in range(len(dataset)**2):
    ax.scatter(x1[i], x2[i], y[i])
    
xx, yy = np.meshgrid(np.arange(-3,3,0.1), np.arange(-3,3,0.1))
z = xx*(w[0][0].item()) + yy*(w[0][1].item()) + b.item()

ax.plot_surface(xx, yy, z)

plt.show()