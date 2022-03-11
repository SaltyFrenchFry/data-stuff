# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 21:47:34 2022

@author: lixin
"""

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd

directory = ""
csv_file = "Obesity Data Random.csv"
dataPath = os.path.join(directory, csv_file)

data_name = pd.read_csv(dataPath)

xy = data_name[["YearStart", "Income", "Data_Value"]].to_numpy()

x = data_name[["YearStart", "Income"]].to_numpy()
y = data_name["Data_Value"].to_numpy()

for i in range(x.shape[0]):
    x[i, 0] = float(x[i, 0] - 2010)
    text = x[i, 1]
    
    if(text == "Less than $15,000"):
        x[i, 1] = float(1.0)
    elif(text == "$15,000 - $24,999"):
        x[i, 1] = float(2.0)
    elif(text == "$25,000 - $34,999"):
        x[i, 1] = float(3.0)
    elif(text == "$35,000 - $49,999"):
        x[i, 1] = float(4.0)
    elif(text == "$50,000 - $74,999"):
        x[i, 1] = float(5.0)
    elif(text == "$75,000 or greater"):
        x[i, 1] = float(6.0)
    
    
pp = torch.zeros(x.shape[0], 2)

for i in range(x.shape[0]):
    pp[i,0] = x[i, 0]
    pp[i,1] = x[i, 1]
    
print(pp)
print(pp.dtype)

print()
print(y)
print(y.dtype)

mom = torch.zeros(y.shape[0])
for i in range(y.shape[0]):
    mom[i] = y[i]
print(mom)
print(mom.dtype)