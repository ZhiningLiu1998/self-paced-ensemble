# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:22:56 2019
@author: Zhining Liu
mailto: znliu19@mails.jlu.edu.cn / zhining.liu@outlook.com
"""

from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score,
    precision_recall_curve, 
    auc, 
    roc_curve, 
    average_precision_score, 
    matthews_corrcoef,
    )
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def load_covtype_dataset(subset=0.1, random_state=None):
    '''Load & Split training/test covtype dataset.'''
    print ('\nDataset used: \t\tForest covertypes from UCI ({:.1%} random subset)'.format(subset))
    X, y = datasets.fetch_covtype(return_X_y=True)
    y = make_binary_classification_target(y, 7, verbose=True)
    X, y = imbalance_random_subset(
        X, y, size=0.1, random_state=random_state)
    X_train, X_test, y_train, y_test = imbalance_train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test

def make_binary_classification_target(y, pos_label, verbose=False):
    '''Turn multi-class targets into binary classification targets.'''
    pos_idx = (y==pos_label)
    y[pos_idx] = 1
    y[~pos_idx] = 0
    if verbose:
        print ('Positive target:\t{}'.format(pos_label))
        print ('Imbalance ratio:\t{:.3f}'.format((y==0).sum()/(y==1).sum()))
    return y

def imbalance_train_test_split(X, y, test_size, random_state=None):
    '''Train/Test split that guarantee same class distribution between split datasets.'''
    X_maj = X[y==0]; y_maj = y[y==0]
    X_min = X[y==1]; y_min = y[y==1]
    X_train_maj, X_test_maj, y_train_maj, y_test_maj = train_test_split(
        X_maj, y_maj, test_size=test_size, random_state=random_state)
    X_train_min, X_test_min, y_train_min, y_test_min = train_test_split(
        X_min, y_min, test_size=test_size, random_state=random_state)
    X_train = np.concatenate([X_train_maj, X_train_min])
    X_test = np.concatenate([X_test_maj, X_test_min])
    y_train = np.concatenate([y_train_maj, y_train_min])
    y_test = np.concatenate([y_test_maj, y_test_min])
    return  X_train, X_test, y_train, y_test

def imbalance_random_subset(X, y, size, random_state=None):
    '''Get random subset while guarantee same class distribution.'''
    _, X, _, y = imbalance_train_test_split(X, y, 
        test_size=size, random_state=random_state)
    return X, y

def auc_prc(label, y_pred):
    '''Compute AUCPRC score.'''
    return average_precision_score(label, y_pred)

def f1_optim(label, y_pred):
    '''Compute optimal F1 score.'''
    y_pred = y_pred.copy()
    prec, reca, _ = precision_recall_curve(label, y_pred)
    f1s = 2 * (prec * reca) / (prec + reca)
    return max(f1s)

def gm_optim(label, y_pred):
    '''Compute optimal G-mean score.'''
    y_pred = y_pred.copy()
    prec, reca, _ = precision_recall_curve(label, y_pred)
    gms = np.power((prec*reca), 0.5)
    return max(gms)

def mcc_optim(label, y_pred):
    '''Compute optimal MCC score.'''
    mccs = []
    for t in range(100):
        y_pred_b = y_pred.copy()
        y_pred_b[y_pred_b < 0+t*0.01] = 0
        y_pred_b[y_pred_b >= 0+t*0.01] = 1
        mcc = matthews_corrcoef(label, y_pred_b)
        mccs.append(mcc)
    return max(mccs)

def precision_at_recall(label, y_pred, recall):
    '''Compute precision at recall.'''
    prec, reca, _ = precision_recall_curve(label, y_pred)
    idx = np.searchsorted(-reca, -recall, 'right')
    return prec[idx - 1]

def recall_at_precision(label, y_pred, precision):
    '''Compute recall at precision.'''
    prec, reca, _ = precision_recall_curve(label, y_pred)
    idx = np.searchsorted(prec, precision, 'right')
    return reca[idx]

class Error(Exception):
    '''Simple exception.'''
    pass