from MIMIC_BenchMark_Data import MIMIC_TrainTest_loader
import torch
import torch.nn as nn
from generate_TS_data import *
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import kurtosis,skew
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.preprocessing import StandardScaler
from IPython.display import display, HTML
import statistics
from torch import nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, batch_size, time_sequence, num_layers=2, dropout_p=0.3):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.time_sequence = time_sequence
        self.drop = nn.Dropout(dropout_p)
        self.num_layers = num_layers
        self.num_classes = output_dim
        bidirectional = True
        if bidirectional == True:
            self.num_directions = 2
        else:
            self.num_directions = 1
            
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=bidirectional)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * self.num_directions, output_dim)
        self.softmax = nn.LogSoftmax(1)

    def init_hidden(self):
        # weights and biasses
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        
        #some lstm expert got empty batch
        batch_got = input.shape[0]
        if batch_got == 0:
            return input.view(-1, self.num_classes)
            
        #reshape input from [batch_size, time * input_dim]
        #to                 [batch_size, time, input_dim]
        input = input.view(-1, self.time_sequence, self.input_dim)
        
        #reshape input from [batch_size, time, input_dim]
        #to                 [time, batch_size, input_dim]
        input = input.permute(1, 0, 2)
        
        # Forward pass through LSTM layer
        # [input_size, batch_size, hidden_dim] => shape of lstm_out => [seq_len, batch_size, num_directions * hidden_size]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim). 
        #lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, self.input_dim))
        #lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, input_size))
        lstm_out, (hidden, cell) = self.lstm(input)
        
        # combine both directions
        combined = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        lstm_out = self.drop(combined)
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        #y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        y_pred = self.linear(lstm_out)
        
        
        #y_pred = self.softmax(lstm_out[-1].view(self.batch_size, -1))
        y_pred = self.softmax(y_pred)
        
        
        #return y_pred.view(-1)
        return y_pred