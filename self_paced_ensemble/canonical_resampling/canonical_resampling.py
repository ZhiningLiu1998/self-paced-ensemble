# -*- coding: utf-8 -*-
"""
A resampling-based classifier for imbalanced classification.
15 resampling algorithms are included: 
'RUS', 'CNN', 'ENN', 'NCR', 'Tomek', 'ALLKNN', 'OSS',
'NM', 'CC', 'SMOTE', 'ADASYN', 'BorderSMOTE', 'SMOTEENN', 
'SMOTETomek', 'ORG'.

The implementation of these resampling algorithms is based on `imblearn`.
Please refer to https://github.com/scikit-learn-contrib/imbalanced-learn.
"""

# Created on Sun Jan 13 14:32:27 2019
# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

from imblearn.under_sampling import (
    ClusterCentroids, 
    NearMiss, 
    RandomUnderSampler, 
    EditedNearestNeighbours, 
    AllKNN, 
    TomekLinks, 
    OneSidedSelection, 
    CondensedNearestNeighbour, 
    NeighbourhoodCleaningRule,
)
from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE,
)
from imblearn.combine import (
    SMOTEENN, SMOTETomek,
)

from sklearn.tree import DecisionTreeClassifier as DT


SUPPORT_RESAMPLING = ['RUS', 'CNN', 'ENN', 'NCR', 'Tomek', 'ALLKNN', 'OSS',
                'NM', 'CC', 'SMOTE', 'ADASYN', 'BorderSMOTE', 'SMOTEENN', 
                'SMOTETomek', 'ORG']


class Error(Exception):
    pass

class ResampleClassifier(object):
    '''
    Re-sampling methods for imbalance classification, based on imblearn python package.
    imblearn url: https://github.com/scikit-learn-contrib/imbalanced-learn
    Hyper-parameters:
        base_estimator : scikit-learn classifier object
            optional (default=DecisionTreeClassifier)
            The base estimator used for training after re-sampling
    '''
    def __init__(self, base_estimator=DT()):
        self.base_estimator = base_estimator
    
    def predict(self, X):
        return self.base_estimator.predict(X)
    
    def fit(self, X, y, by, random_state=None, visualize=False):
        '''
        by: String
            The method used to perform re-sampling
            support: ['RUS', 'CNN', 'ENN', 'NCR', 'Tomek', 'ALLKNN', 'OSS',
                'NM', 'CC', 'SMOTE', 'ADASYN', 'BorderSMOTE', 'SMOTEENN', 
                'SMOTETomek', 'ORG']
        '''
        if by == 'RUS':
            sampler = RandomUnderSampler(random_state=random_state)
        elif by == 'CNN':
            sampler = CondensedNearestNeighbour(random_state=random_state)
        elif by == 'ENN':
            sampler = EditedNearestNeighbours()
        elif by == 'NCR':
            sampler = NeighbourhoodCleaningRule()
        elif by == 'Tomek':
            sampler = TomekLinks()
        elif by == 'ALLKNN':
            sampler = AllKNN()
        elif by == 'OSS':
            sampler = OneSidedSelection(random_state=random_state)
        elif by == 'NM':
            sampler = NearMiss()
        elif by == 'CC':
            sampler = ClusterCentroids(random_state=random_state)
        elif by == 'SMOTE':
            sampler = SMOTE(random_state=random_state)
        elif by == 'ADASYN':
            sampler = ADASYN(random_state=random_state)
        elif by == 'BorderSMOTE':
            sampler = BorderlineSMOTE(random_state=random_state)
        elif by == 'SMOTEENN':
            sampler = SMOTEENN(random_state=random_state)
        elif by == 'SMOTETomek':
            sampler = SMOTETomek(random_state=random_state)
        elif by == 'ORG':
            sampler = None
        else:
            raise Error('Unexpected \'by\' type {}'.format(by))
        
        if by != 'ORG':
            X_train, y_train = sampler.fit_resample(X, y)
        else:
            X_train, y_train = X, y
        self.base_estimator.fit(X_train, y_train)