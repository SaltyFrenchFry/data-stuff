# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 21:02:24 2022

@author: Evan Wang, Sudhish Swain, Alisha Wu, Sophia Zeng, Jerry Jing
"""
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

directory = ""
csv_file = "Education_Data_Random.csv"
dataPath = os.path.join(directory, csv_file)

data_name = pd.read_csv(dataPath)

x = data_name[["YearStart", "Education"]].to_numpy()
y = data_name["Data_Value"].to_numpy()

for i in range(x.shape[0]):
    x[i, 0] = float(x[i, 0] - 2010)
    text = x[i, 1]
    
    if(text == "Less than high school"):
        x[i, 1] = float(1.0)
    elif(text == "High school graduate"):
        x[i, 1] = float(2.0)
    elif(text == "Some college or technical school"):
        x[i, 1] = float(3.0)
    elif(text == "College graduate"):
        x[i, 1] = float(4.0)


class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out

class Data2D(Dataset):
    def __init__(self, train = True):
        if(train == True):
            self.x = torch.zeros(int(x.shape[0]*0.8), 2)
            
            for i in range(int(x.shape[0]*0.8)):
                self.x[i,0] = x[i, 0]
                self.x[i,1] = x[i, 1]
            
            self.y = torch.zeros(int(x.shape[0]*0.8), 1)
            
            for i in range(int(x.shape[0]*0.8)):
                self.y[i,0] = y[i]
            
            self.len = int(x.shape[0]*0.8)
        else:
            self.x = torch.zeros(x.shape[0]-int(x.shape[0]*0.8), 2)
            
            j = 0
            for i in range(x.shape[0]-int(x.shape[0]*0.8)):
                self.x[j,0] = x[i, 0]
                self.x[j,1] = x[i, 1]
                j += 1;
            
            self.y = torch.zeros(x.shape[0]-int(x.shape[0]*0.8), 1)
            
            j = 0
            for i in range(x.shape[0]-int(x.shape[0]*0.8)):
                self.y[j, 0] = y[i]
                j += 1
            
            self.len = x.shape[0]-int(x.shape[0]*0.8)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len

criterion = nn.MSELoss()

dataset = Data2D()
valData = Data2D(train = False)

epochs = 50

batchSize = [1, 2, 4, 8, 16, 32, 64, 128, 256]
learningRates = range(1, 21)

validationError = torch.zeros([len(learningRates), len(batchSize)])

testError = torch.zeros([len(learningRates), len(batchSize)])

MODELS = [[0 for x in range(len(batchSize))] for y in range(len(learningRates))]

min_loss = 1000000
bestLearningRate = 0
bestBatchSize = 0

for j,batch_size in enumerate(batchSize):
    
    trainloader = DataLoader(dataset = dataset, batch_size = batch_size)
    
    for i,learning_rate in enumerate(learningRates):
        model = LR(2, 1)
        
        learning_rate /= 1000
        
        optimizer = optim.SGD(model.parameters(), lr = learning_rate)
        
        min_loss = 1000000
        
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
        
        model_best = LR(2, 1)
        model_best.load_state_dict(torch.load("best_model.pt"))
        
        MODELS[i][j] = model_best
        
        yhat = model_best(dataset.x)
        loss = criterion(yhat, dataset.y)
        testError[i, j] = loss.item()
        
        yhat = model_best(valData.x)
        loss = criterion(yhat, valData.y)
        validationError[i, j] = loss.item()

for i in range(len(batchSize)):
    plt.plot(np.array(learningRates),testError.numpy()[:,i], label = "train cost")
    plt.plot(np.array(learningRates),validationError.numpy()[:,i], label = "validation cost")
    plt.xlabel("Learning rate multiplied by 1000")
    plt.ylabel("Loss")
    plt.title("Loss for batch size of " + str(batchSize[i]))
    plt.legend()
    plt.show()

bestLoss = validationError[0, 0]
for i in range(validationError.shape[0]):
    for j in range(validationError.shape[1]):
        if(validationError[i,j] < bestLoss):
            bestLearningRate = i
            bestBatchSize = j
            bestLoss = validationError[i,j]


model_best = MODELS[bestLearningRate][bestBatchSize]

n = 0
for param in model_best.parameters():
    if(n == 0):
        w = param.data
        n = 1
    else:
        b = param.data

print(testError)
print(validationError)
print(w)
print(b)
print(bestLearningRate)
print(bestBatchSize)
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i in range(len(dataset)):
    ax.scatter(dataset.x[i,0], dataset.x[i,1], dataset.y[i])
for i in range(len(valData)):
    ax.scatter(valData.x[i,0], valData.x[i,1], valData.y[i])
    
xx, yy = np.meshgrid(np.arange(1,10,0.1), np.arange(1,4,0.1))
z = xx*(w[0][0].item()) + yy*(w[0][1].item()) + b.item()

ax.plot_surface(xx, yy, z)

ax.set_xlabel('Year')
ax.set_ylabel('Education')
ax.set_zlabel('Obesity Percent')

plt.show()