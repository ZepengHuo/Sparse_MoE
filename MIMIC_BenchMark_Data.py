# %%
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

# %%
normal_values_dict = {
    'Capillary refill rate': 0,
    'Diastolic blood pressure': 59,
    'Fraction inspired oxygen': 0.21,
    'Glascow coma scale eye opening': '4 Spontaneously',
    'Glascow coma scale motor response': '6 Obeys Commands',
    'Glascow coma scale total': 15,
    'Glascow coma scale verbal response': '5 Oriented',
    'Glucose': 128,
    'Heart Rate': 86,
    'Height': 170,
    'Mean blood pressure': 77,
    'Oxygen saturation': 98,
    'Respiratory rate': 19,
    'Systolic blood pressure': 118,
    'Temperature': 36.6,
    'Weight': 81,
    'pH': 7.4
}

# %%
eye_opening_impute_dict = {
    'None': 1,
    '1 No Response': 1, 
    'No Response': 1, 
    'No response': 1,
    '2 To pain': 2,
    'To pain': 2,
    'To Pain': 2,
    '3 To speech': 3,
    'To speech': 3,
    'To Speech': 3,
    '4 Spontaneously': 4,
    'Spontaneously': 4
}

# %%
motor_response_impute_dict = {
    '1 No Response': 1,
    'No Response': 1,
    'No response': 1,
    '2 Abnorm extensn': 2,
    'Abnorm extension': 2,
    'Abnormal extension': 2,
    '3 Abnorm flexion': 3,
    'Abnorm Flexion': 3,
    'Abnormal Flexion': 3,
    '4 Flex-withdraws': 4,
    'Flex-withdraws': 4,
    '5 Localizes Pain': 5,
    'Localizes Pain': 5,
    '6 Obeys Commands': 6,
    'Obeys Commands': 6,    
}

# %%
verbal_response_impute_dict = {
    '1.0 ET/Trach': 1,
    'ET/Trach': 1,
    'No Response': 1,
    'No response': 1,
    '1 No Response': 1,
    'No Response-ETT': 1,  
    '2 Incomp sounds': 2,
    'Incomprehensible sounds': 2,
    '3 Inapprop words': 3,
    'Inappropriate Words': 3,
    '4 Confused': 4,
    'Confused': 4,
    '5 Oriented': 5,
    'Oriented': 5,
    'Spontaneously': 5
    
}

# %%
#categorical features

categorial_ls = ['Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale verbal response']

# %%
def timeSeriesFeature(df, index, one_csv):
    df = df.drop(columns=['Hours'])
    
    all_features = []

    
    col_ls = df.columns
    for col in col_ls:
        one_TS_feature_ls = []
        
        one_TS = df[col].values
        
        #if col not in categorial_ls:

        #add mean feature   
        try:
            one_TS_feature_ls.append(np.mean(one_TS))
        except:
            print(index, one_csv)
            display(df)

        #add std feature
        one_TS_feature_ls.append(np.std(one_TS))

        #add max feature
        one_TS_feature_ls.append(np.amax(one_TS))

        #add min feature
        one_TS_feature_ls.append(np.amin(one_TS))

        #add kurtosis feature
        try:
            one_TS_feature_ls.append(kurtosis(one_TS))
        except:
            
            one_TS_feature_ls.append(0)

        #add skewness feature
        one_TS_feature_ls.append(skew(one_TS))
        
        
        '''
        #categorical features
        else:
            majority = np.unique(one_TS)[0]
            majority = majority.split(' ')
            majority = majority[0]
            try:
                majority = int(majority)
            except:
                majority = 4
                
            one_TS_feature_ls.append(majority)
         '''   
        
        all_features.extend(one_TS_feature_ls)
        
    return np.array(all_features)
        

# %%
def raw_48h_data(one_df_imputed, interval=1):
    one_df_empty = one_df_imputed[0:0]
    last_one_df_mean = one_df_empty
    
    i = 1
    for i in range(0, 48):

        last_i = i-interval

        one_df_mean = pd.DataFrame(one_df_imputed[(one_df_imputed['Hours'] < i) & (one_df_imputed['Hours'] > last_i)].mean())
        one_df_mean = one_df_mean.T
        
        # see if 1h interval has data
        if one_df_mean.dropna(axis=0).shape[0] > 0:
            one_df_empty = pd.concat([one_df_empty, one_df_mean], axis = 0)
            
        # see last 1h has data     
        elif last_one_df_mean.dropna(axis=0).shape[0] > 0:
            one_df_empty = pd.concat([one_df_empty, last_one_df_mean], axis = 0)
            
        #if first hour no data    
        else:
            one_df_empty.loc[len(one_df_empty)] = 0
            
        last_one_df_mean = one_df_mean
    
    one_df_empty = one_df_empty[1:]
    one_df_empty = one_df_empty.drop(columns=['Hours'])
    return one_df_empty

# %%
def normalize_std(one_df_imputed):
    for col in one_df_imputed.columns:
        if col not in categorial_ls:
            sc = StandardScaler()
            sc.fit(one_df_imputed[col].values.reshape(-1, 1))
            one_df_imputed[col] = sc.transform(one_df_imputed[col].values.reshape(-1, 1))
            
    return one_df_imputed

# %%
def categorical_feature(one_df_imputed):
    one_df_imputed['Glascow coma scale eye opening'] = one_df_imputed['Glascow coma scale eye opening'].replace(eye_opening_impute_dict)
    one_df_imputed['Glascow coma scale motor response'] = one_df_imputed['Glascow coma scale motor response'].replace(motor_response_impute_dict)
    one_df_imputed['Glascow coma scale verbal response'] = one_df_imputed['Glascow coma scale verbal response'].replace(verbal_response_impute_dict)

    one_df_imputed['Glascow coma scale eye opening'] = one_df_imputed['Glascow coma scale eye opening'].astype('int32')
    one_df_imputed['Glascow coma scale motor response'] = one_df_imputed['Glascow coma scale motor response'].astype('int32')
    one_df_imputed['Glascow coma scale verbal response'] = one_df_imputed['Glascow coma scale verbal response'].astype('int32')

    one_df_imputed['Glascow coma scale total'] = one_df_imputed['Glascow coma scale eye opening'] + one_df_imputed['Glascow coma scale motor response'] + one_df_imputed['Glascow coma scale verbal response']

    one_df_imputed['Glascow coma scale total'] = one_df_imputed['Glascow coma scale total'].astype('int32')   
    
    return one_df_imputed

# %%
def MIMIC_TrainTest_loader(device_num, frac, normalize, extracted):
    #change to sys root
    os.chdir('/data/datasets/mimic3-benchmarks/data/')
    
    
    # decompensation  length-of-stay  phenotyping  in-hospital-mortality  multitask
    task = 'in-hospital-mortality'
    
    #change to sys root
    os.chdir('/data/datasets/mimic3-benchmarks/data/' + task)
    
    df_train = pd.read_csv('train_listfile.csv')
    #df_val = pd.read_csv('val_listfile.csv')
    df_test = pd.read_csv('test_listfile.csv')
    
    df_train = df_train.sample(frac=frac, replace=False, random_state=1)
    df_test = df_test.sample(frac=frac, replace=False, random_state=1)
    
    y_train = df_train['y_true'].values
    #y_val = df_val['y_true'].values
    y_test = df_test['y_true'].values
    
    
    X_train = []
    ############## training data ##############
    for index, row in df_train.iterrows():
            
        one_csv = row['stay']
        
        rePath = 'train/'
        one_df = pd.read_csv(rePath + one_csv)

        #forward imputation
        one_df_ffill = one_df.fillna(method='ffill')

        #impute with normal value
        one_df_imputed = one_df_ffill.fillna(value=normal_values_dict)
        
        #normalize with std
        if normalize == True:
            one_df_imputed = normalize_std(one_df_imputed)
            
        #categorical feature engineering
        one_df_imputed = categorical_feature(one_df_imputed)
        
        #extract feature from one df OR raw data
        if extracted == True:
            features_ls = timeSeriesFeature(one_df_imputed, index, one_csv)
        else:
            features_ls = raw_48h_data(one_df_imputed, interval=1).values.tolist()

        X_train.append(features_ls)

    X_train = np.array(X_train)

    
    #get input size and num of classes
    input_size = X_train.shape[-1]
    num_classes = np.unique(y_train).size
    
    
    X_test = []
    ############### testing data ##############
    for index, row in df_test.iterrows():
        
        one_csv = row['stay']
        rePath = 'test/'
        one_df = pd.read_csv(rePath + one_csv)

        #forward imputation
        one_df_ffill = one_df.fillna(method='ffill')

        #impute with normal value
        one_df_imputed = one_df_ffill.fillna(value=normal_values_dict)
        
        #normalize with std
        if normalize == True:
            one_df_imputed = normalize_std(one_df_imputed)
        
        #categorical feature engineering
        one_df_imputed = categorical_feature(one_df_imputed)

        #extract feature from one df OR raw data
        if extracted == True:
            features_ls = timeSeriesFeature(one_df_imputed, index, one_csv)
        else:
            features_ls = raw_48h_data(one_df_imputed, interval=1).values.tolist()

        X_test.append(features_ls)
    
    X_test = np.array(X_test)

    
    '''
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
        
        X_train, X_test = X_train.type(torch.double), X_test.type(torch.double)
        y_train, y_test = y_train.type(torch.long), y_test.type(torch.long)
        
        
    train = data_utils.TensorDataset(X_train, y_train)
    #train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    test = data_utils.TensorDataset(X_test, y_test)
    #test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)
    
    
    
    #num_classes = y_data.shape[1]
    '''
    
    return X_train, y_train, X_test, y_test, input_size, num_classes     

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
