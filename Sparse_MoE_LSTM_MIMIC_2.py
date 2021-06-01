# %%
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
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
import seaborn as sn
from moe import MoE
from validate import validate
from sklearn.manifold import TSNE
import seaborn as sns
import pickle
from datetime import datetime
np.random.seed(0) 

# %%
time_now =datetime.now().strftime("%Y-%m-%d %H:%M").replace(' ', '_')

# %%
device_num = 4

use_cuda = True
if device_num:
    use_cuda = torch.cuda.is_available()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device_num)

device = torch.device("cuda:{}".format(str(device_num)) if use_cuda else "cpu")

# %%
def ForBCELoss_convert(y_true, device_num):
    y_true_expand = np.zeros((y_true.shape[0], 2))
    
    for idx in range(y_true.shape[0]):
        y_true_expand[idx][y_true[idx]] = 1
    
    y_true_expand = torch.FloatTensor(y_true_expand).cuda(device_num)
    y_true_expand = y_true_expand.type(torch.float)
    
    return y_true_expand

# %%
def ProbaToPred(y_pred):
    a = nn.functional.softmax(y_pred, 1)
    pred = a.max(1).indices.data.to('cpu').numpy()
    
    return pred

# %%
#change data type directly in code. Dummy way
def Reload_TensorType(X_train, y_train, X_test, y_test, device_num):

    if device_num:
        X_train = torch.FloatTensor(X_train).cuda(device_num)
        y_train = torch.tensor(y_train).cuda(device_num)
        X_test = torch.FloatTensor(X_test).cuda(device_num)
        y_test = torch.tensor(y_test).cuda(device_num)
        
        X_train, X_test = X_train.type(torch.float), X_test.type(torch.float)
        y_train, y_test = y_train.type(torch.long), y_test.type(torch.long)
        
    else:
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)
        X_test = torch.tensor(X_test)
        y_test = torch.tensor(y_test)
        
        X_train, X_test = X_train.type(torch.float), X_test.type(torch.float)
        y_train, y_test = y_train.type(torch.long), y_test.type(torch.long)
        
        
    train = data_utils.TensorDataset(X_train, y_train)
    #train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    test = data_utils.TensorDataset(X_test, y_test)
    #test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)
    
    return train, test

# %%
#[batch_size, time, input_dim] => [batch_size, time * input_dim]
def dataShape_forLSTM(X_train, y_train, X_test, y_test, expert_type):
    
    if expert_type == 'lstm' and len(X_train.shape) == 3:
        time_sequence = X_train.shape[1]
        input_size = X_train.shape[2] * time_sequence
        
        X_train = X_train.reshape(len(X_train), -1)
        X_test = X_test.reshape(len(X_test), -1)
    
    return X_train, y_train, X_test, y_test, time_sequence, input_size

# %%
def backHome():
    home = os.path.expanduser('~')
    path = '/Research/Interpretability/Sparsely_Gated_Mixture_of_Experts/mixture-of-experts/'
    os.chdir(home+path)

# %%
def MIMIC_bechmark_processed(frac):
    with open('./mimic3-benchmarks-repo/mortality_train', 'rb') as handle:
        train_file = pickle.load(handle)

    with open('./mimic3-benchmarks-repo/mortality_val', 'rb') as handle:
        test_file = pickle.load(handle)

    X_train, y_train = train_file[0], np.array(train_file[1])
    X_test, y_test = test_file[0], np.array(test_file[1])
    
    train_len = len(X_train)
    test_len = len(X_test)
    train_crop = int(train_len * frac)
    test_crop = int(test_len * frac)
    
    train_idx = np.random.randint(train_len, size=train_crop)
    test_idx = np.random.randint(test_len, size=test_crop)
    
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    X_test, y_test = X_test[test_idx], y_test[test_idx]
    
    num_classes = len(np.unique(y_train))
    
    return X_train, y_train, X_test, y_test, num_classes, train_idx, test_idx
# %%
"""
# Below model setup
"""
for batch_size in [32, 64, 128, 256]:
    for num_experts in [30, 50, 100, 150]:
        for k in np.array(num_experts * np.array([0.2, 0.4, 0.6]), dtype='int'):
            for weight_ratio in [3, 4, 5, 6, 7]:
                for hidden_size in [32, 64, 128, 256]:
                    for dropout_p in [0.1, 0.2, 0.3]:

                        print('='*50)
                        print(batch_size, num_experts, k, weight_ratio, hidden_size, dropout_p)
                        #####################
                        # Set dataloader parameters
                        #####################

                        expert_type = 'lstm'
                        device_num = device_num
                        frac = 0.01
                        normalize = False
                        extracted = False
                        #batch_size=32

                        # %%
                        #####################
                        # Set LSTM parameters 
                        #####################

                        # Network params
                        # If `per_element` is True, then LSTM reads in one timestep at a time.
                        per_element = False
                        '''
                        if per_element:
                            lstm_input_size = 1
                        else:
                            lstm_input_size = input_size
                        '''
                        # size of hidden layers
                        #h1 = 32
                        
                        num_layers = 2
                        learning_rate = 1e-5
                        num_epochs = 200
                        dtype = torch.float
                        #dropout_p = 0.3

                        # %%
                        #####################
                        # Set MoE layer parameters 
                        #####################


                        #num_experts = 50
                        #hidden_size = 128
                        #k = 20  #k: an integer - how many experts to use for each batch element
                        lr = 0.00001
                        checkLoss_every = 100
                        validate_every = 100
                        binary = True
                        #weight_ratio = 3

                        # %%
                        # get MIMIC data
                        X_train, y_train, X_test, y_test, num_classes, train_idx, test_idx = MIMIC_bechmark_processed(frac)
                        output_dim = num_classes
                        
                        with open('./mimic3-benchmarks-repo/mortality_train.pkl', 'rb') as handle:
                            train_file = pickle.load(handle)

                        with open('./mimic3-benchmarks-repo/mortality_val.pkl', 'rb') as handle:
                            test_file = pickle.load(handle)       
                        
                        train_file = np.array(train_file[2])[train_idx]
                        test_file = np.array(test_file[2])[test_idx]
                        with open('./result_dump/train_file_%s.pkl'%time_now, 'wb') as f0:

                            pickle.dump(train_file, f0)

                        # %%
                        with open('./result_dump/test_file_%s.pkl'%time_now, 'wb') as f0:

                            pickle.dump(test_file, f0)
                        
                        
                        # %%
                        # go back home directory
                        backHome()

                        # %%
                        # reshape for LSTM expert, get time_sequence
                        X_train, y_train, X_test, y_test, time_sequence, input_size = dataShape_forLSTM(X_train, y_train, X_test, y_test, expert_type)

                        # %%
                        ############################
                        #put np.array into Tensor, get Tensor type right
                        ######## caveat: re-run may kill the kernel! ########
                        ############################
                        train, test = Reload_TensorType(X_train, y_train, X_test, y_test, device_num)

                        # %%
                        #put data into loader
                        train_loader = data_utils.DataLoader(train, batch_size=batch_size, drop_last=True, shuffle=False)
                        test_loader = data_utils.DataLoader(test, batch_size=batch_size*10, drop_last=True, shuffle=False)

                        # %%
                        """
                        # instantiate the MoE layer
                        """

                        # %%
                        model = MoE(expert_type, input_size, num_classes, num_experts, hidden_size, batch_size, time_sequence=time_sequence,
                                    num_layers=2, dropout_p=0.3 , k=k, noisy_gating=True, device_num=device_num)

                        if device_num:
                            model = model.cuda(device_num)

                        # %%
                        """
                        # instantiate lstm model
                        """

                        # %%
                        #class imbalance re-weight
                        weight = torch.tensor([1, weight_ratio]).type(torch.float32).cuda(device_num)

                        ########## loss function ################    
                        #loss_fn = torch.nn.MSELoss(size_average=False)
                        #loss_fn = torch.nn.NLLLoss(weight=weight, size_average=False)
                        #loss_fn = torch.nn.NLLLoss(size_average=False)

                        #cross entropy loss seems to be best for LSTM
                        loss_fn = nn.CrossEntropyLoss(weight=weight)

                        #if BCELoss or BCEWithLogitsLoss => both pred and true both be size of [batch_size]
                        #loss_fn = nn.BCEWithLogitsLoss(size_average=False)
                        #loss_fn = nn.BCELoss(size_average=False)
                        #########################################

                        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

                        # %%
                        #####################
                        # Train model
                        #####################

                        hist = np.zeros(num_epochs)


                        for t in range(num_epochs):
                            Gates_ls = []
                            Y_ls = []

                            model = model.train()
                            for X_train, y_train in train_loader:

                                #[bath_size, time, input_dime] -> [bath_size, time * input_dime]
                                #X_train = X_train.view(len(X_train), -1)

                                # Initialise hidden state
                                # Don't do this if you want your LSTM to be stateful
                                #model.hidden = model.init_hidden()

                                # Forward pass
                                #y_pred = model(X_train)
                                y_pred, aux_loss, gates = model(X_train)
                                Gates_ls.append(gates)
                                Y_ls.append(y_train)

                                loss = loss_fn(y_pred, y_train)

                                total_loss = loss + aux_loss

                                # Zero out gradient, else they will accumulate between epochs
                                optimiser.zero_grad()

                                # Backward pass
                                #loss.backward()
                                total_loss.backward()

                                # Update parameters
                                optimiser.step()


                            if t % 100 == 0:
                                print("Epoch ", t, "Loss: ", loss.item())
                            hist[t] = loss.item()

                            if t % validate_every == 0:
                                if binary:
                                    f1, acc, auroc, aucpr = validate(model, test_loader, binary)
                                    print("Validation Results- F1 score: {:.3f}, accuracy: {:.3f}, auroc: {:.3f}, auc_pr: {:.3f}".format(f1, acc, auroc, aucpr))        
                                else:
                                    f1, acc = validate(model, test_loader, binary)
                                    print("Validation Results - F1 score: {:.3f}, accuracy: {:.3f}".format(f1, acc))

                        #####################
                        # Plot preds and performance
                        #####################

                        y_pred = y_pred.cpu()
                        y_train = y_train.cpu()

                        plt.plot(y_pred.clone().detach().numpy(), label="Preds")
                        plt.plot(y_train.clone().detach().numpy(), label="Data")
                        plt.legend()
                        plt.show()

                        plt.plot(hist, label="Training loss")
                        plt.legend()
                        plt.show()

                        # %%
                        assert os.path.exists('./result_dump/')
                        with open('./result_dump/gate_%s.pkl'%time_now, 'wb') as f0:

                            pickle.dump(Gates_ls, f0)

                        # %%
                        with open('./result_dump/y_%s.pkl'%time_now, 'wb') as f0:

                            pickle.dump(Y_ls, f0)

                        # %%
                        model.eval()
                        f1_score_ls = []
                        accuracy_ls = []
                        valid_loss = []
                        all_true = np.array([])
                        all_pred = np.array([])

                        for data, target in test_loader:
                            #permute data for LSTM
                            #data = data.permute(1, 0, 2)

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
                        df_cm = confusion_matrix(all_true, all_pred)
                        plt.figure(figsize = (6,4))
                        sn.heatmap(df_cm, annot=True)
                        plt.xlabel('Prediction')
                        plt.ylabel('Groundtruth')
                        plt.savefig('./result_dump/confusion_matrix_%s.png'%(time_now))
                        plt.show()
                        
                        del model