# %%
import torch
#reproducibility
torch.manual_seed(1)

import os
from torch.utils.data import Dataset, DataLoader
from moe import MoE
from torch.optim import Adam
from torch import nn
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sn
from validate import validate
import statistics

# %%
"""
# 1. Parameters setting
"""

# %%
device_num = 0
# False or 0 - 7


use_cuda = True
if device_num:
    use_cuda = torch.cuda.is_available()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device_num)

device = torch.device("cuda:{}".format(str(device_num)) if use_cuda else "cpu")

# %%
num_experts = 50
hidden_size = 100
batch_size = 512
k = 20  #k: an integer - how many experts to use for each batch element
lr = 0.00001
epochs = 600
checkLoss_every = 100
validate_every = 100
binary = True

# %%
"""
# 2. get dataset
"""

# %%
from MIMIC_BenchMark_Data import MIMIC_TrainTest_loader

train_loader, test_loader, input_size, num_classes = MIMIC_TrainTest_loader(device_num, batch_size)

# %%
"""
# 3. instantiate the MoE layer
"""

# %%
model = MoE(input_size, num_classes, num_experts, hidden_size, k=k, noisy_gating=True, device_num=device_num)

if device_num:
    model = model.cuda(device_num)

# %%
"""
# 4. train model on GPU
"""

# %%
optim = Adam(model.parameters(), lr=lr)
#optim = torch.optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, 25]).type(torch.float32))
#loss_fn = nn.NLLLoss(weight=torch.tensor([1, 10]).type(torch.float32))
#loss_fn = nn.NLLLoss()
#loss_fn = nn.BCEWithLogitsLoss()


for epoch in range(epochs):
    #optim.zero_grad()
    for x_train, y_train in train_loader:
        
        model = model.train()
        model = model.double()

        y_hat, aux_loss = model(x_train)

        #for TOPCAT binary cls
        #y_hat = torch.squeeze(y_hat)
        #y_train = y_train.type(torch.float32)
        #y_train = y_train.unsqueeze(1)
        
        loss = loss_fn(y_hat, y_train)

        total_loss = loss + aux_loss

        total_loss.backward()

        optim.step()
        
    if epoch % checkLoss_every == 0:
        print('epoch=', epoch, end=' ')
        print("Training Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()), end='  ')
        
    if epoch % validate_every == 0:
        if binary:
            f1, acc, auroc = validate(model, test_loader, binary)
            print("Validation Results - F1 score: {:.3f}, accuracy: {:.3f}, auroc: {:.3f}".format(f1, acc, auroc))        
        else:
            f1, acc = validate(model, test_loader, binary)
            print("Validation Results - F1 score: {:.3f}, accuracy: {:.3f}".format(f1, acc))
    

# %%
"""
# 5. Testing
"""

# %%
model.eval()
f1_score_ls = []
accuracy_ls = []
valid_loss = []
all_true = np.array([])
all_pred = np.array([])

for data, target in test_loader:
    output = model(data)
    a = nn.functional.softmax(output[0], 1)
    pred = a.max(1).indices.data.to('cpu').numpy()
    true = target.data.to('cpu').numpy()
    
    all_true = np.append(all_true, true)
    all_pred = np.append(all_pred, pred)
    
    f1Score = f1_score(true, pred, average='macro')
    Accuracy = accuracy_score(true, pred)
    
    f1_score_ls.append(f1Score)
    accuracy_ls.append(Accuracy)
    
    #loss = loss_fn(output, target)
    #valid_loss.append(loss.item())

# %%
print(statistics.mean(f1_score_ls))
print(statistics.mean(accuracy_ls))

# %%
df_cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize = (6,4))
sn.heatmap(df_cm, annot=True)
plt.xlabel('Prediction')
plt.ylabel('Groundtruth')

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
"""
# different weight for imbalance class
"""