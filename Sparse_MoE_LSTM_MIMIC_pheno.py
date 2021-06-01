# %%
from __future__ import absolute_import
from __future__ import print_function

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
from torch import nn, optim
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import seaborn as sn
from moe import MoE
from validate import validate
from sklearn.manifold import TSNE
import seaborn as sns
import pickle
from datetime import datetime
from torchsummary import summary
from sklearn import metrics
np.random.seed(0) 

# %%
def Time_Now():
    return datetime.now().strftime("%Y-%m-%d %H:%M").replace(' ', '_')

# %%
device_num = 6

use_cuda = True
if device_num:
    use_cuda = torch.cuda.is_available()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device_num)

device = torch.device("cuda:{}".format(str(device_num)) if use_cuda else "cpu")

# %%
def save_torch_model(model_spec, model, optimiser, hist, hist_val, epoch):
    state = {
        'state_dict': model.state_dict(),
        'optimiser': optimiser.state_dict(),
        'hist': hist,
        'hist_val': hist_val,
        'epoch': epoch
    }
    torch.save(state, 'Models/' + model_spec + '.pt')

# %%
def print_metrics_multilabel(y_true, predictions, verbose=1):
    y_true = np.array(y_true)
    predictions = np.array(predictions)

    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")

    if verbose:
        print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return ave_auc_micro, ave_auc_macro
    # return {"auc_scores": auc_scores,
    #         "ave_auc_micro": ave_auc_micro,
    #         "ave_auc_macro": ave_auc_macro,
    #         "ave_auc_weighted": ave_auc_weighted}

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
def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

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
def MIMIC_bechmark_mortality_processed(frac):

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
def MIMIC_bechmark_phenotype_processed(frac):
    with open('./mimic3-benchmarks-repo/pheno_train', 'rb') as handle:
        train_file = pickle.load(handle)

    with open('./mimic3-benchmarks-repo/pheno_test', 'rb') as handle:
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
    
    num_classes = y_train.shape[1]
    
    return X_train, y_train, X_test, y_test, num_classes, train_idx, test_idx

# %%
"""
# Below model setup
"""
for batch_size in [32, 64, 128, 256]:
    for num_experts in [30, 50, 100, 150]:
        for k in np.array(num_experts * np.array([0.2, 0.4, 0.6]), dtype='int'):
            for hidden_size in [32, 64, 128, 256]:
                for dropout_p in [0.1, 0.2, 0.3]:

                    print('='*50)
                    print('batch_size={}, num_experts={}, k={}, hidden_size={}, dropout_p={}'.format(batch_size, num_experts, k, hidden_size, dropout_p))
                    
                    # %%
                    task = 'pheno'

                    #batch_size = 64 
                    #num_experts = 50 
                    #k = 20    #k: an integer - how many experts to use for each batch element
                    weight_ratio = 6 
                    #hidden_size = 64 
                    #dropout_p = 0.3
                    noisy_gating = True

                    # %%
                    #####################
                    # Set dataloader parameters
                    #####################

                    expert_type = 'lstm'
                    device_num = device_num
                    frac = 0.1
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
                    learning_rate = 1e-6
                    start_epoch = 1
                    num_epochs = 300
                    dtype = torch.float
                    #dropout_p = 0.3
                    timestep = 1.0  # 1 hour interval

                    # %%
                    #####################
                    # Set MoE layer parameters 
                    #####################

                    #k = 20  
                    checkLoss_every = 100
                    validate_every = 100
                    binary = False
                    #weight_ratio = 3

                    # %%
                    if task == 'mortality':
                        # get MIMIC mortality data

                        X_train, y_train, X_test, y_test, num_classes, train_idx, test_idx = MIMIC_bechmark_mortality_processed(frac)

                    elif task == 'pheno':

                        # get MIMIC phenotype data
                        X_train, y_train, X_test, y_test, num_classes, train_idx, test_idx = MIMIC_bechmark_phenotype_processed(frac)

                    # %%
                    output_dim = num_classes

                    # reshape for LSTM expert, get time_sequence
                    X_train, y_train, X_test, y_test, time_sequence, input_size = dataShape_forLSTM(X_train, y_train, X_test, y_test, expert_type)


                    ############################
                    #put np.array into Tensor, get Tensor type right
                    ######## caveat: re-run may kill the kernel! ########
                    ############################
                    train, test = Reload_TensorType(X_train, y_train, X_test, y_test, device_num)

                    #put data into loader
                    train_loader = data_utils.DataLoader(train, batch_size=batch_size, drop_last=True, shuffle=False)
                    test_loader = data_utils.DataLoader(test, batch_size=len(test), drop_last=True, shuffle=False)

                    # %%
                    # go back home directory
                    backHome()

                    # %%
                    """
                    # instantiate the MoE layer
                    """

                    # %%
                    model = MoE(expert_type, input_size, num_classes, num_experts, hidden_size, batch_size, time_sequence=time_sequence,
                                num_layers=2, dropout_p=0.3 , k=k, noisy_gating=noisy_gating, device_num=device_num)

                    if device_num:
                        model = model.cuda(device_num)

                    # %%
                    """
                    # instantiate (single) lstm model
                    """

                    # %%
                    """
                    # optimiser setup
                    """

                    # %%
                    #class imbalance re-weight
                    weight = torch.tensor([1, weight_ratio]).type(torch.float32).cuda(device_num)

                    ########## loss function ################    
                    #loss_fn = torch.nn.MSELoss(size_average=False)
                    #loss_fn = torch.nn.NLLLoss(weight=weight, size_average=False)
                    #loss_fn = torch.nn.NLLLoss(size_average=False)

                    if task == 'mortality':
                        #cross entropy loss seems to be best for LSTM
                        loss_fn = nn.CrossEntropyLoss(weight=weight)
                    elif task == 'pheno':
                        loss_fn = nn.BCEWithLogitsLoss()

                    #if BCELoss or BCEWithLogitsLoss => both pred and true both be size of [batch_size]
                    #loss_fn = nn.BCEWithLogitsLoss(size_average=False)
                    #loss_fn = nn.BCELoss(size_average=False)
                    #########################################

                    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

                    # %%
                    """
                    # load a model or not
                    """

                    # %%
                    # pick newest model's time
                    Model_files = os.listdir('Models')
                    time_ls = []
                    for file in Model_files:
                        if file.endswith('pt'):
                            time_str = file[-19:-3]
                            time_obj = datetime.strptime(time_str ,  "%Y-%m-%d_%H:%M")
                            time_ls.append(time_obj)

                    time_ls.sort()
                    time_ls[-1]
                    timelast_str = time_ls[-1].strftime("%Y-%m-%d %H:%M").replace(' ', '_')
                    for file in Model_files:
                        if file.endswith(timelast_str+'.pt'):
                            Model_path = 'Models/' + file

                    # %%
                    # load a trained model

                    #state = torch.load(Model_path)

                    #model.load_state_dict(state['state_dict'])
                    #optimiser.load_state_dict(state['optimiser'])

                    # %%
                    # loss history
                    hist = np.array([])
                    hist_val = np.array([])

                    # %%
                    # load loss history
                    #hist = state['hist']
                    #hist_val = state['hist_val']
                    #start_epoch = state['epoch']

                    # %%
                    """
                    # training phase
                    """

                    # %%
                    #####################
                    # Train model
                    #####################



                    for t in range(start_epoch, start_epoch + num_epochs):
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

                            loss = loss_fn(y_pred, y_train.float())

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
                        hist = np.append(hist, loss.item())

                        with torch.no_grad():
                            model.eval()
                            model.apply(apply_dropout)


                            for X_test, y_test in test_loader:

                                y_pred, aux_loss, gates = model(X_test)
                                loss = loss_fn(y_pred, y_test.float())
                                hist_val = np.append(hist_val, loss.item())
                                break

                        if t % validate_every == 0:
                            if binary:
                                f1, acc, auroc, aucpr = validate(model, test_loader, binary)
                                print("Validation Results- F1 score: {:.3f}, accuracy: {:.3f}, auroc: {:.3f}, auc_pr: {:.3f}".format(f1,acc,
                                                                                                                                     auroc,aucpr))     
                            else:
                                y_pred = y_pred.cpu().detach().numpy()
                                y_train = y_train.cpu().detach().numpy()
                                y_test = y_test.cpu().detach().numpy()
                                ave_auc_micro, ave_auc_macro = print_metrics_multilabel(y_test, y_pred)

                                #f1, acc = validate(model, test_loader, binary)
                                #print("Validation Results - F1 score: {:.3f}, accuracy: {:.3f}".format(f1, acc))


                    #####################
                    # Plot preds and performance
                    #####################


                    '''
                    y_pred = y_pred.cpu()
                    y_train = y_train.cpu()
                    y_test = y_test.cpu()

                    plt.plot(y_pred.clone().detach().numpy(), label="Preds")
                    plt.plot(y_train.clone().detach().numpy(), label="Data")
                    plt.legend()
                    plt.show()
                    '''


                    plt.plot(hist, label="Training loss")
                    plt.plot(hist_val, label="Validation loss")
                    plt.legend()
                    plt.show()

                    # %%
                    """
                    # save trained model
                    """

                    # %%
                    time_now = Time_Now()
                    model_spec = "task{}_batch{}_#E{}_k{}_type{}_hid{}_drop{}_{}".format(task, batch_size, num_experts, k, expert_type, hidden_size, dropout_p, time_now)

                    # %%
                    save_torch_model(model_spec, model, optimiser, hist, hist_val, t)

                    # %%



                    del model