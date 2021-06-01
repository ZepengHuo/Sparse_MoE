# %%
import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils

# %%
def UCI_TrainTest_loader(device_num, batch_size):


    UCI_ABSpath = '/home/grads/g/guangzhou92/Research/Darpa/UQ_AISTATS/OPPORTUNITY/MIX/preprocessed'

    os.chdir(UCI_ABSpath)

    #info_train, info_test

    df_train = pd.read_csv('info_train.csv', sep=' ')
    df_test = pd.read_csv('info_test.csv', sep=' ')
    
    
    #get train from npy
    ##############slow############

    features = []
    targets = []
    rows, cols = df_train.shape[0], df_train.shape[1]

    for row in range(rows):

        file_name = df_train.iloc[row, 0] + '.npy'
        activity = df_train.iloc[row, 1] - 1
        context = df_train.iloc[row, 2]

        feature_flat = np.load(file_name).flatten()
        feature_flat = feature_flat.tolist()

        features.append(feature_flat)
        targets.append(activity.tolist())     
        
        
    input_size = len(feature_flat)
    num_classes = np.unique(targets).shape[0]   
    
    
    #get test from npy
    ##############slow############

    features_test = []
    targets_test = []
    rows, cols = df_test.shape[0], df_test.shape[1]

    for row in range(rows):

        file_name = df_test.iloc[row, 0] + '.npy'
        activity = df_test.iloc[row, 1] - 1
        context = df_test.iloc[row, 2]

        feature_flat = np.load(file_name).flatten()
        feature_flat = feature_flat.tolist()

        features_test.append(feature_flat)
        targets_test.append(activity.tolist())
        
        
    # data on CPU or GPU
    if device_num:
        features = torch.tensor(features).cuda(device_num)
        targets = torch.tensor(targets).cuda(device_num)
        features_test = torch.tensor(features_test).cuda(device_num)
        targets_test = torch.tensor(targets_test).cuda(device_num)

    else:
        features = torch.tensor(features)
        targets = torch.tensor(targets)
        features_test = torch.tensor(features_test)
        targets_test = torch.tensor(targets_test)

    #load to dataloader
    train = data_utils.TensorDataset(features, targets)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    test = data_utils.TensorDataset(features_test, targets_test)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, input_size, num_classes

# %%
# ipynb-py-convert examples/plot.ipynb examples/plot.py

# %%
