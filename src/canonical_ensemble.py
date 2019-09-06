# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 14:32:27 2019
@author: v-zhinli
mailto: v-zhinli@microsoft.com / zhining.liu@outlook.com
"""

"""
IMPORTANT!
The implementation of SMOTE/SMOTEBoost/RUSBoost was obtained from
imbalanced-algorithms: https://github.com/dialnd/imbalanced-algorithms

We have to stress that, according to our tests, the performance 
of our baseline method implementation is FAR SUPERIOR to the 
implementation in the imblearn package.
imblearn package: https://github.com/scikit-learn-contrib/imbalanced-learn
"""

from collections import Counter

import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.forest import BaseForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from sklearn.utils import check_array
from sklearn.preprocessing import binarize

class SMOTE(object):
    """Implementation of Synthetic Minority Over-Sampling Technique (SMOTE).
    SMOTE performs oversampling of the minority class by picking target 
    minority class samples and their nearest minority class neighbors and 
    generating new samples that linearly combine features of each target 
    sample with features of its selected minority class neighbors [1].
    Parameters
    ----------
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O. Hall, and P. Kegelmeyer. "SMOTE:
           Synthetic Minority Over-Sampling Technique." Journal of Artificial
           Intelligence Research (JAIR), 2002.
    """

    def __init__(self, k_neighbors=5, random_state=None):
        self.k = k_neighbors
        self.random_state = random_state

    def sample(self, n_samples):
        """Generate samples.
        Parameters
        ----------
        n_samples : int
            Number of new synthetic samples.
        Returns
        -------
        S : array, shape = [n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        S = np.zeros(shape=(n_samples, self.n_features))
        # Calculate synthetic samples.
        for i in range(n_samples):
            j = np.random.randint(0, self.X.shape[0])

            # Find the NN for each sample.
            # Exclude the sample itself.
            nn = self.neigh.kneighbors(self.X[j].reshape(1, -1),
                                       return_distance=False)[:, 1:]
            nn_index = np.random.choice(nn[0])

            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]

        return S

    def fit(self, X):
        """Train model based on input data.
        Parameters
        ----------
        X : array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples.
        """
        self.X = X
        self.n_minority_samples, self.n_features = self.X.shape

        # Learn nearest neighbors.
        self.neigh = NearestNeighbors(n_neighbors=self.k + 1)
        self.neigh.fit(self.X)

        return self

class SMOTEBoost(AdaBoostClassifier):
    """Implementation of SMOTEBoost.
    SMOTEBoost introduces data sampling into the AdaBoost algorithm by
    oversampling the minority class using SMOTE on each boosting iteration [1].
    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer.
           "SMOTEBoost: Improving Prediction of the Minority Class in
           Boosting." European Conference on Principles of Data Mining and
           Knowledge Discovery (PKDD), 2003.
    """

    def __init__(self,
                 n_samples=100,
                 k_neighbors=5,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.n_samples = n_samples
        self.algorithm = algorithm
        self.smote = SMOTE(k_neighbors=k_neighbors,
                           random_state=random_state)

        super(SMOTEBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing SMOTE during each boosting step.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.
        Returns
        -------
        self : object
            Returns self.
        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # SMOTE step.
            X_min = X[np.where(y == self.minority_target)]
            self.smote.fit(X_min)
            X_syn = self.smote.sample(self.n_samples)
            y_syn = np.full(X_syn.shape[0], fill_value=self.minority_target,
                            dtype=np.int64)

            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
            sample_weight_syn[:] = 1. / X.shape[0]

            # print ('Boosting Iter: {} n_train: {} n_smote: {}'.format(
            #     iboost, len(X_min), len(y_syn)))

            # Combine the original and synthetic samples.
            X = np.vstack((X, X_syn))
            y = np.append(y, y_syn)

            # Combine the weights.
            sample_weight = \
                np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))

            # X, y, sample_weight = shuffle(X, y, sample_weight,
            #                              random_state=random_state)

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination.
            if sample_weight is None:
                print('sample_weight: {}'.format(sample_weight))
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            # if estimator_error == 0:
            #     print('error: {}'.format(estimator_error))
            #     break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                print('sample_weight_sum: {}'.format(sample_weight_sum))
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self

class RandomUnderSampler(object):
    """Implementation of random undersampling (RUS).
    Undersample the majority class(es) by randomly picking samples with or
    without replacement.
    Parameters
    ----------
    with_replacement : bool, optional (default=True)
        Undersample with replacement.
    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly selected
        from the majority class.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    """

    def __init__(self, with_replacement=True, return_indices=False,
                 random_state=None):
        self.return_indices = return_indices
        self.with_replacement = with_replacement
        self.random_state = random_state

    def sample(self, n_samples):
        """Perform undersampling.
        Parameters
        ----------
        n_samples : int
            Number of samples to remove.
        Returns
        -------
        S : array, shape = [n_majority_samples - n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        if self.n_majority_samples <= n_samples:
            n_samples = self.n_majority_samples

        idx = np.random.choice(self.n_majority_samples,
                            #    size=self.n_majority_samples - n_samples,
                               size=self.n_minority_samples,
                               replace=self.with_replacement)

        if self.return_indices:
            return (self.X_maj[idx], idx)
        else:
            return self.X_maj[idx]

    def fit(self, X_maj, X_min):
        """Train model based on input data.
        Parameters
        ----------
        X : array-like, shape = [n_majority_samples, n_features]
            Holds the majority samples.
        """
        self.X_maj = X_maj
        self.X_min = X_min
        self.n_majority_samples, self.n_features = self.X_maj.shape
        self.n_minority_samples = self.X_min.shape[0]

        return self

import pandas as pd

class RUSBoost(AdaBoostClassifier):
    """Implementation of RUSBoost.
    RUSBoost introduces data sampling into the AdaBoost algorithm by
    undersampling the majority class using random undersampling (with or
    without replacement) on each boosting iteration [1].
    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.
    min_ratio : float (default=1.0)
        Minimum ratio of majority to minority class samples to generate.
    with_replacement : bool, optional (default=True)
        Undersample with replacement.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] C. Seiffert, T. M. Khoshgoftaar, J. V. Hulse, and A. Napolitano.
           "RUSBoost: Improving Classification Performance when Training Data
           is Skewed". International Conference on Pattern Recognition
           (ICPR), 2008.
    """

    def __init__(self,
                 n_samples=100,
                 min_ratio=1.0,
                 with_replacement=True,
                 base_estimator=None,
                 n_estimators=10,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.n_samples = n_samples
        self.min_ratio = min_ratio
        self.algorithm = algorithm
        self.rus = RandomUnderSampler(with_replacement=with_replacement,
                                      return_indices=True,
                                      random_state=random_state)

        super(RUSBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        minority_target : int
            Minority class label.
        Returns
        -------
        self : object
            Returns self.
        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Random undersampling step.
            X_maj = X[np.where(y != self.minority_target)]
            X_min = X[np.where(y == self.minority_target)]
            self.rus.fit(X_maj, X_min)
            # self.rus.fit(X_maj)

            n_maj = X_maj.shape[0]
            n_min = X_min.shape[0]
            if n_maj - self.n_samples < int(n_min * self.min_ratio):
                self.n_samples = n_maj - int(n_min * self.min_ratio)
            X_rus, X_idx = self.rus.sample(self.n_samples)

            # print ('Boosting Iter: {} X_maj: {} X_rus: {} X_min: {}'.format(
            #     iboost, len(X_maj), len(X_rus), len(X_min)))

            y_rus = y[np.where(y != self.minority_target)][X_idx]
            y_min = y[np.where(y == self.minority_target)]

            sample_weight_rus = \
                sample_weight[np.where(y != self.minority_target)][X_idx]
            sample_weight_min = \
                sample_weight[np.where(y == self.minority_target)]

            # Combine the minority and majority class samples.
            X_train = np.vstack((X_rus, X_min))
            y_train = np.append(y_rus, y_min)

            # Combine the weights.
            sample_weight_train = \
                np.append(sample_weight_rus, sample_weight_min).reshape(-1, 1)
            sample_weight_train = \
                np.squeeze(normalize(sample_weight_train, axis=0, norm='l1'))

            # Boosting step.
            _, estimator_weight_train, estimator_error = self._boost(
                iboost,
                X_train, y_train,
                sample_weight_train,
                random_state)
            
            # print(self.estimators_)
            y_predict_proba = self.estimators_[-1].predict_proba(X)
            y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                        axis=0)
            # Instances incorrectly classified
            incorrect = y_predict != y
            # Error fraction
            estimator_error = np.mean(
                np.average(incorrect, weights=sample_weight, axis=0))
            n_classes = self.n_classes_
            classes = self.classes_
            y_codes = np.array([-1. / (n_classes - 1), 1.])
            y_coding = y_codes.take(classes == y[:, np.newaxis])
            estimator_weight = (-1. * self.learning_rate
                    * ((n_classes - 1.) / n_classes)
                    * (y_coding * (y_predict_proba)).sum(axis=1))
            # print(y_predict_proba, y_coding, np.log(y_predict_proba))
            if not iboost == self.n_estimators - 1:
                # Only boost positive weights
                sample_weight *= np.exp(estimator_weight * ((sample_weight > 0) | (estimator_weight < 0)))
                # print (np.exp(estimator_weight * ((sample_weight > 0) | (estimator_weight < 0))))
            # Early termination.
            if sample_weight is None:
                print('sample_weight: {}'.format(sample_weight))
                break

            self.estimator_weights_[iboost] = estimator_weight_train
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            # if estimator_error == 0:
                # print('error: {}'.format(estimator_error))
                # break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                print('sample_weight_sum: {}'.format(sample_weight_sum))
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self

import pandas as pd
from imblearn.over_sampling import SMOTE as SMOTE_IMB
from sklearn.tree import DecisionTreeClassifier as DT

class SMOTEBagging():
    def __init__(self,
                 n_samples=100,
                 min_ratio=1.0,
                 with_replacement=True,
                 base_estimator=None,
                 n_estimators=10,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):
        self.n_estimators = n_estimators
        self.model_list = []
    
    def fit(self, X, y):
        self.model_list = []
        df = pd.DataFrame(X); df['label'] = y
        df_maj = df[df['label']==0]; n_maj = len(df_maj)
        df_min = df[df['label']==1]; n_min = len(df_min)
        cols = df.columns.tolist(); cols.remove('label')
        for ibagging in range(self.n_estimators):
            b = min(0.1*((ibagging%10)+1), 1)
            train_maj = df_maj.sample(frac=b, replace=True)
            train_min = df_min.sample(frac=b, replace=True)
            # train_maj = df_maj.sample(frac=1/self.n_estimators, replace=True)
            # train_min = df_min.sample(frac=1/self.n_estimators, replace=True)
            # train_maj = df_maj.sample(n=n_min, replace=True)
            # train_min = df_min.sample(frac=1/self.n_estimators, replace=True)
            df_k = train_maj.append(train_min)
            X_train, y_train = SMOTE_IMB(k_neighbors=min(5, len(train_min)-1)).fit_resample(df_k[cols], df_k['label'])
            # print ('Bagging Iter: {} |b: {:.1f}|n_train: {}|n_smote: {}'.format(
            #     ibagging, b, len(y_train), len(y_train)-len(df_k)))
            model = DT().fit(X_train, y_train)
            self.model_list.append(model)
        return self
    
    def predict_proba(self, X):
        y_pred = np.array([model.predict(X) for model in self.model_list]).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1-y_pred, y_pred, axis=1)
        return y_pred
    
    def predict(self, X):
        y_pred_binarazed = binarize(self.predict_proba(X)[:,1].reshape(1,-1), threshold=0.5)[0]
        return y_pred_binarazed

import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DT

class UnderBagging():
    def __init__(self,
                 n_samples=100,
                 min_ratio=1.0,
                 with_replacement=True,
                 base_estimator=None,
                 n_estimators=10,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):
        self.n_estimators = n_estimators
        self.model_list = []
    
    def fit(self, X, y):
        self.model_list = []
        df = pd.DataFrame(X); df['label'] = y
        df_maj = df[df['label']==0]; n_maj = len(df_maj)
        df_min = df[df['label']==1]; n_min = len(df_min)
        cols = df.columns.tolist(); cols.remove('label')
        for ibagging in range(self.n_estimators):
            train_maj = df_maj.sample(n=n_min, replace=True)
            train_min = df_min
            # print ('Bagging Iter: {} X_maj: {} X_rus: {} X_min: {}'.format(
            #     ibagging, len(df_maj), len(train_maj), len(train_min)))
            df_k = train_maj.append(train_min)
            X_train, y_train = df_k[cols], df_k['label']
            model = DT().fit(X_train, y_train)
            self.model_list.append(model)
        return self
    
    def predict_proba(self, X):
        y_pred = np.array([model.predict(X) for model in self.model_list]).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1-y_pred, y_pred, axis=1)
        return y_pred
    
    def predict(self, X):
        y_pred_binarazed = binarize(self.predict_proba(X)[:,1].reshape(1,-1), threshold=0.5)[0]
        return y_pred_binarazed


from sklearn.base import clone
class BalanceCascade():
    """
    The implementation of BalanceCascade.
    Hyper-parameters:
        base_estimator : scikit-learn classifier object
            optional (default=DecisionTreeClassifier)
            The base estimator from which the ensemble is built.
        n_estimators:       Number of iterations / estimators
        k_bins:             Number of hardness bins
    """
    def __init__(self, base_estimator=DT(), n_estimators=10, random_seed=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.model_list = []
        # Will be set in the fit function
        self.feature_cols = None

    def _fit_baselearner(self, df_train):
        model = clone(self.base_estimator)
        return model.fit(df_train[self.feature_cols], df_train['label'])

    def fit(self, X, y, print_log=False, visualize=False):
        # Initialize majority & minority set
        df = pd.DataFrame(X); df['label'] = y
        df_maj = df[y==0]; n_maj = df_maj.shape[0]
        df_min = df[y==1]; n_min = df_min.shape[0]
        self.feature_cols = df.columns.tolist()
        self.feature_cols.remove('label')

        ir = n_min / n_maj
        keep_fp_rate = np.power(ir, 1/(self.n_estimators-1))

        # Algorithm start
        for ibagging in range(1, self.n_estimators):
            df_train = df_maj.sample(n=n_min).append(df_min)
            if visualize:
                df_train.plot.scatter(x=0, y=1, s=3, c='label', colormap='coolwarm', title='Iter {} training set'.format(ibagging))
            # print ('Cascade Iter: {} X_maj: {} X_rus: {} X_min: {}'.format(
            #     ibagging, len(df_maj), len(df_min), len(df_min)))
            self.model_list.append(self._fit_baselearner(df_train))
            # drop "easy" majority samples
            df_maj['pred_proba'] = self.predict(df_maj[self.feature_cols])
            df_maj = df_maj.sort_values(by='pred_proba', ascending=False)[:int(keep_fp_rate*len(df_maj)+1)]
        return self
    
    def predict_proba(self, X):
        y_pred = np.array([model.predict(X) for model in self.model_list]).mean(axis=0)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]
        if y_pred.shape[1] == 1:
            y_pred = np.append(1-y_pred, y_pred, axis=1)
        return y_pred
    
    def predict(self, X):
        y_pred_binarazed = binarize(self.predict_proba(X)[:,1].reshape(1,-1), threshold=0.5)[0]
        return y_pred_binarazed
