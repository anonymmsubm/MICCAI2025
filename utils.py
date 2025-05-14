import json
import torch
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

def load_pickle_file(file_path):
    with open(file_path, 'rb') as pickle_file:
        dataframe = pickle.load(pickle_file)
    return dataframe

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, type):  
            return obj.__name__
        return json.JSONEncoder.default(self, obj)
    
def model_eval(model, weights, test_loader, device):
    try:
        model.load_state_dict(weights)
    except:
        model.load_state_dict(weights["state_dict"])
    model.to(device)
    model.eval()
    labels_validation = []
    pred = []
    for batch in test_loader:
        with torch.no_grad():
            if len(batch) == 2:
                seq, labels = batch
                labels = labels.to(device)
                y_pred = model(seq.to(device))
            else: 
                seq, patch, mask, labels = batch
                seq = seq.to(device)
                patch = patch.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                y_pred = model(seq, patch.flatten(start_dim=2), mask)
            pred.extend(y_pred.cpu().numpy())
            labels_validation.extend(labels.cpu().numpy())
    pred = np.array(pred)
    labels_validation = np.array(labels_validation)
    
    # Compute ROC AUC Score
    roc_auc = roc_auc_score(labels_validation, pred)
    print(f'ROC AUC Score: {roc_auc:.4f}')

    # Compute ROC Curve
    fpr, tpr, thresholds = roc_curve(labels_validation, pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    # Apply optimal threshold
    prediction = (pred > optimal_threshold).astype(int)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(labels_validation, prediction)
    print("\nConfusion Matrix:\n", conf_matrix)

    # Compute classification report
    print("\nClassification Report:\n", classification_report(labels_validation, prediction, digits=4))

    # Compute Sensitivity (Recall) and Specificity
    tn, fp, fn, tp = conf_matrix.ravel()
    recall = tp / (tp + fn)  # Sensitivity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Print additional metrics
    print(f'Accuracy:  {accuracy_score(labels_validation, prediction):.4f}')
    print(f'F1 Score:  {f1_score(labels_validation, prediction):.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall (Sensitivity): {recall:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print('=============')
    
