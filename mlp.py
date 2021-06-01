# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_size, num_layers=2, dropout_p=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        #self.fc3 = nn.Linear(hidden_size, 1)
        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(hidden_size)
        self.log_soft = nn.LogSoftmax(1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        
        out = self.fc1(x)
        out = self.relu(out)
        #out = self.bn(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.log_soft(out)
        #out = self.sigmoid(out)
        return out
