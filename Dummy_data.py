# %%
import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils

# %%
def dummy_data(batch_size, input_size, num_classes):
    # dummy input
    x = torch.rand(batch_size, input_size)

    # dummy target
    y = torch.randint(num_classes, (batch_size, 1)).squeeze(1)
    return x,y

# %%
def Dummy_TrainTest_loader(device_num, batch_size, input_size, num_classes):
    
    
    
    x_train, y_train = dummy_data(batch_size*8, input_size, num_classes)
    x_eval, y_eval = dummy_data(batch_size*2, input_size, num_classes)

    x_train, y_train = x_train.cuda(device_num), y_train.cuda(device_num)
    x_eval, y_eval = x_eval.cuda(device_num), y_eval.cuda(device_num)
    
    
    train = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    test = data_utils.TensorDataset(x_eval, y_eval)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)
    
    
    return train_loader, test_loader

# %%
# ipynb-py-convert examples/plot.ipynb examples/plot.py

# %%
