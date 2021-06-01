# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.impute import SimpleImputer

# %%
def TOPCAT_TrainTest_loader(device_num, batch_size):
    
    outcome_cols = [
    'death',
    'cvd_death',
    'time_death',
    'anyhosp',
    'time_anyhosp',
    'hfhosp',
    'time_hfhosp',
    'abortedca',
    'time_abortedca',
    'mi',
    'time_mi',
    'stroke',
    'time_stroke',
    'primary_ep',
    'time_primary_ep'
    ]
    
    
    os.chdir('/data/datasets/topcat/py_cleaned_data')
    
    df = pd.read_csv('TOPCAT_final_2_25_2020.csv')
    
    
    
    df_X = df.drop(columns=outcome_cols)
    df_y = df['cvd_death']
    
    x_data = df_X.values
    y_data = df_y.values

    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_data = imp.fit_transform(x_data)
    
    
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5)
    
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
        #y_train, y_test = y_train.type(torch.float32), y_test.type(torch.float32)
        y_train, y_test = y_train.type(torch.long), y_test.type(torch.long)
        
    else:
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)
        X_test = torch.tensor(X_test)
        y_test = torch.tensor(y_test)

    
    
    train = data_utils.TensorDataset(X_train, y_train)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    test = data_utils.TensorDataset(X_test, y_test)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)
    
    
    input_size = x_data.shape[1]
    #num_classes = y_data.shape[1]
    num_classes = np.unique(y_data).size
    
    
    return train_loader, test_loader, input_size, num_classes

# %%
