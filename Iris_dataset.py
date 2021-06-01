# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils

# %%
def one_hot_label(target):
    num_classes = np.unique(target).size
    y = torch.zeros(target.shape[0], num_classes)
    
    y[range(y.shape[0]), target]=1
    
    return y

# %%
def Iris_TrainTest_loader(device_num, batch_size):
    iris = load_iris()
    
    x_data=iris.data
    y_data=iris.target
    #y_data = one_hot_label(y_data)
    
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    
    
    # data on CPU or GPU
    if device_num:
        X_train = torch.FloatTensor(X_train).cuda(device_num)
        y_train = torch.tensor(y_train).cuda(device_num)
        X_test = torch.FloatTensor(X_test).cuda(device_num)
        y_test = torch.tensor(y_test).cuda(device_num)
        
        #X_train = X_train.clone().detach().requires_grad_(True)
        #y_train = y_train.clone().detach().requires_grad_(True)
        #X_test = X_test.clone().detach().requires_grad_(True)
        #y_test = y_test.clone().detach().requires_grad_(True)
        
        X_train, X_test = X_train.type(torch.float32), X_test.type(torch.float32)
        y_train, y_test = y_train.type(torch.long), y_test.type(torch.long)
        
    else:
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)
        X_test = torch.tensor(X_test)
        y_test = torch.tensor(y_test)

    
        X_train, X_test = X_train.type(torch.double), X_test.type(torch.double)
        y_train, y_test = y_train.type(torch.long), y_test.type(torch.long)
    
    
    train = data_utils.TensorDataset(X_train, y_train)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    test = data_utils.TensorDataset(X_test, y_test)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)
    
    
    input_size = x_data.shape[1]
    #num_classes = y_data.shape[1]
    num_classes = np.unique(y_data).size
    
    
    return train_loader, test_loader, input_size, num_classes