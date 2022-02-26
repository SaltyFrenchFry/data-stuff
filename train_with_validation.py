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
import numpy as np

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
        self.x = torch.arange(-3, 3, 0.1).view(-1,1)
        self.y = 3*self.x+1 + 0.3*torch.randn(self.x.size())
        self.len = self.x.shape[0]
        
        if train:
            self.y[0] = 1
            self.y[50:60] = 20
        else:
            pass
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len
    
dataset = Data()
validationData = Data(train = False)
trainingData = Data()

trainloader = DataLoader(dataset=trainingData, batch_size=5)

w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)

epochs = 10
learningRates = [0.0001, 0.001, 0.01, 0.1, 1]

validationError = torch.zeros(len(learningRates))

testError = torch.zeros(len(learningRates))

for i,learning_rate in enumerate(learningRates):
    
    model = LR(1, 1)
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    min_loss = 10000
    
    for epoch in range(epochs):
        
        total = 0
        isFirst = True
        
        for x,y in trainloader:
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total += loss.item()
        
        if(total < min_loss or isFirst):
            value = epoch
            min_loss = total
            torch.save(model.state_dict(),"best_model_" + str(learning_rate)+".pt")
            
        isFirst = False
                
    yhat = model(trainingData.x)
    loss = criterion(yhat, trainingData.y)
    testError[i] = loss.item()
    
    yhat = model(validationData.x)
    loss = criterion(yhat, validationData.y)
    validationError[i] = loss.item()

plt.semilogx(np.array(learningRates),testError.numpy(), label = "train cost")
plt.semilogx(np.array(learningRates),validationError.numpy(), label = "validation cost")
plt.legend()
plt.show()

for learning_rate in learningRates:
    model = LR(1,1)
    model.load_state_dict(torch.load("best_model_" + str(learning_rate)+".pt"))
    yhat = model(validationData.x)
    plt.plot(validationData.x.numpy(), yhat.detach().numpy(),label = "lr:" + str(learning_rate))

plt.plot(trainingData.x.numpy(),trainingData.y.numpy(), "or", label = "validation data")
plt.plot(validationData.x.numpy(),validationData.y.numpy(), "ob", label = "training data")
plt.legend()
plt.show()

plt.plot(trainingData.x.numpy(),trainingData.y.numpy(), "or", label = "training data")
plt.plot(validationData.x.numpy(),validationData.y.numpy(), "ob", label = "training data")
plt.legend()
plt.show()