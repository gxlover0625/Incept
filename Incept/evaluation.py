import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score

def best_mapping(true_labels, pred_labels):
    true_label_set = np.unique(true_labels)
    pred_label_set = np.unique(pred_labels)
    
    matrix = np.zeros((len(true_label_set), len(pred_label_set)), dtype=int)
    
    for i, true_label in enumerate(true_label_set):
        for j, pred_label in enumerate(pred_label_set):
            matrix[i, j] = np.sum((true_labels == true_label) & (pred_labels == pred_label))
    
    row_ind, col_ind = linear_sum_assignment(-matrix)
    
    mapping = {}
    for i, j in zip(row_ind, col_ind):
        mapping[pred_label_set[j]] = true_label_set[i]
    
    remapped_labels = np.array([mapping[label] for label in pred_labels])
    
    return remapped_labels

def acc(true_labels, pred_labels):
    remapped_labels = best_mapping(true_labels, pred_labels)
    return accuracy_score(true_labels, remapped_labels)

def nmi(true_labels, pred_labels):
    return normalized_mutual_info_score(true_labels, pred_labels)

def ari(true_labels, pred_labels):
    return adjusted_rand_score(true_labels, pred_labels)

class Evaluator:
    def __init__(self, metrics = ["acc", "nmi", "ari"]):
        self.metrics = metrics
        self.funcs = {
            "acc": acc,
            "nmi": nmi,
            "ari": ari
        }
    
    def eval(self, true_labels, pred_labels):
        results = {}
        for m in self.metrics:
            results[m] = self.funcs[m](true_labels, pred_labels)
        return results