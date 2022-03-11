import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms

directory = ""
csv_file = "data.csv"
dataPath = os.path.join(directory, csv_file)

data_name = pd.read_csv(dataPath)

num = data_name.to_numpy()


class SameerData(Dataset):
    def __init__(self, csv_file, data_dir, transform = None):
        self.transform = transform
        
        self.data_dir = data_dir
        data_dircsv_file = os.path.join(self.data_dir, csv_file)
        
        self.data_name = pd.read_csv(data_dircsv_file)
        
        self.len = self.data_name.shape[0]
        
        num = data_name.to_numpy()
        
        self.x = torch.from_numpy(num[:,0]);
        self.y = torch.from_numpy(num[:,1]).float();
        
    
    def __len__(self):
        return(self.len)

    def __getitem__(self, index):
        
        sample = self.x[index], self.y[index];
        
        if(self.transform):
            sample = self.transform(sample)
        
        return sample

class MultTransform(object):
    def __init__(self, mult):
        self.mult = mult
    
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        y *= self.mult
        sample = x, y
        return(sample)
    
class RoundTransform(object):
    def __init__(self, decimal):
        self.decimal = decimal
        
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        y = torch.FloatTensor([round(y.item(), self.decimal)])
        sample = x, y
        return(sample)
    


dataTransform = transforms.Compose([MultTransform(0.01), RoundTransform(2)])

dataset = SameerData(csv_file = csv_file, data_dir = directory, transform = dataTransform)

for x in range(len(dataset)):
    print(dataset[x])
    
plt.plot(dataset.x, dataset.y)
plt.show()