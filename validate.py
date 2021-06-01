import statistics
from torch import nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc

def validate(model, val_loader, binary=True):
    #binary classifier or not
    
    f1_score_ls = []
    accuracy_ls = []
    auroc_score_ls = []
    auc_pr_ls = []
    all_true = np.array([])
    all_pred = np.array([])
    model = model.eval()
    for data, target in val_loader:
        output = model(data)
        a = nn.functional.softmax(output[0], 1)
        pred = a.max(1).indices.data.to('cpu').numpy()
        
        #1 here means the class 1's proba
        pred_proba = a[:, 1].data.to('cpu').numpy()
        #pred_proba = a.data.to('cpu').numpy()
        
        true = target.data.to('cpu').numpy()

        all_true = np.append(all_true, true)
        all_pred = np.append(all_pred, pred)
        
        
        if binary:
            f1Score = f1_score(true, pred, average='macro')
            Accuracy = accuracy_score(true, pred)
            auroc_score = roc_auc_score(true, pred_proba)
            precision, recall, thresholds = precision_recall_curve(true, pred_proba)
            auc_pr = auc(recall, precision)
                
            f1_score_ls.append(f1Score)
            accuracy_ls.append(Accuracy)
            auroc_score_ls.append(auroc_score)
            auc_pr_ls.append(auc_pr)
            
        else:
            f1Score = f1_score(true, pred, average='macro')
            Accuracy = accuracy_score(true, pred)
            

            f1_score_ls.append(f1Score)
            accuracy_ls.append(Accuracy)

        
    if binary:
        return statistics.mean(f1_score_ls), statistics.mean(accuracy_ls), statistics.mean(auroc_score_ls), statistics.mean(auc_pr_ls)
    else:
        return statistics.mean(f1_score_ls), statistics.mean(accuracy_ls)