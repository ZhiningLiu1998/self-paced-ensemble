# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:32:27 2019
@author: ZhiningLiu1998
mailto: znliu19@mails.jlu.edu.cn / zhining.liu@outlook.com

A self-paced Ensemble (SPE) Classifier for binary class-imbalanced learning.
    
Self-paced Ensemble (SPE) is an ensemble learning framework for massive highly 
imbalanced classification. It is an easy-to-use solution to class-imbalanced 
problems, features outstanding computing efficiency, good performance, and wide 
compatibility with different learning models.
"""

# %%

# import packages
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
from collections import Counter

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_random_state, check_is_fitted, column_or_1d, check_array
from sklearn.utils.multiclass import check_classification_targets

from joblib import Parallel, effective_n_jobs
import functools
from functools import update_wrapper
from contextlib import contextmanager as contextmanager

_global_config = {
    'assume_finite': bool(os.environ.get('SKLEARN_ASSUME_FINITE', False)),
    'working_memory': int(os.environ.get('SKLEARN_WORKING_MEMORY', 1024)),
    'print_changed_only': True,
    'display': 'text',
}

def get_config():
    return _global_config.copy()

def set_config(assume_finite=None, working_memory=None,
               print_changed_only=None, display=None):
    if assume_finite is not None:
        _global_config['assume_finite'] = assume_finite
    if working_memory is not None:
        _global_config['working_memory'] = working_memory
    if print_changed_only is not None:
        _global_config['print_changed_only'] = print_changed_only
    if display is not None:
        _global_config['display'] = display

@contextmanager
def config_context(**new_config):
    old_config = get_config().copy()
    set_config(**new_config)

    try:
        yield
    finally:
        set_config(**old_config)

def _parallel_predict_proba(estimators, estimators_features, X, n_classes):
    """Private function used to compute (proba-)predictions within a job."""
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))

    for estimator, features in zip(estimators, estimators_features):
        if hasattr(estimator, "predict_proba"):
            proba_estimator = estimator.predict_proba(X[:, features])

            if n_classes == len(estimator.classes_):
                proba += proba_estimator

            else:
                proba[:, estimator.classes_] += \
                    proba_estimator[:, range(len(estimator.classes_))]

        else:
            # Resort to voting
            predictions = estimator.predict(X[:, features])

            for i in range(n_samples):
                proba[i, predictions[i]] += 1

    return proba

def delayed(function):
    """Decorator used to capture the arguments of a function."""
    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        return _FuncWrapper(function), args, kwargs
    return delayed_function

class _FuncWrapper:
    """"Load the global configuration before calling the function."""
    def __init__(self, function):
        self.function = function
        self.config = get_config()
        update_wrapper(self, self.function)

    def __call__(self, *args, **kwargs):
        with config_context(**self.config):
            return self.function(*args, **kwargs)

def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


class SelfPacedEnsembleClassifier(BaseEnsemble, ClassifierMixin):
    """A self-paced Ensemble (SPE) Classifier for binary class-imbalanced learning.
    
    Self-paced Ensemble (SPE) is an ensemble learning framework for massive highly 
    imbalanced classification. It is an easy-to-use solution to class-imbalanced 
    problems, features outstanding computing efficiency, good performance, and wide 
    compatibility with different learning models.

    Parameters
    ----------
    base_estimator : object, optional (default=sklearn.Tree.DecisionTreeClassifier())
        The base estimator to fit on self-paced under-sampled subsets of the dataset. 
        NO need to support sample weighting. 
        Built-in `fit()`, `predict()`, `predict_proba()` methods are required.

    hardness_func : function, optional 
        (default=`lambda y_true, y_pred: np.absolute(y_true-y_pred)`)
        User-specified classification hardness function
            Parameters:
                y_true: 1-d array-like, shape = [n_samples] 
                y_pred: 1-d array-like, shape = [n_samples] 
            Returns:
                hardness: 1-d array-like, shape = [n_samples]

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    k_bins : int, optional (default=10)
        The number of hardness bins that were used to approximate hardness distribution.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    n_jobs : int, default=None
        The number of jobs to run in parallel for :meth:`predict`. 
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` 
        context. ``-1`` means using all processors. See 
        :term:`Glossary <n_jobs>` for more details.

    random_state : int / RandomState instance / None, optional (default=None)
        If int, random_state is the seed used by the random number generator; 
        If RandomState instance, random_state is the random number generator; 
        If None, the random number generator is the RandomState instance used by 
        `numpy.random`.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    base_estimator : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimator
        The collection of fitted base estimators.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4,
    ...                         n_informative=3, n_redundant=0,
    ...                         n_classes=2, random_state=0, 
    ...                         shuffle=False)
    >>> clf = SelfPacedEnsembleClassifier(
    ...         base_estimator=DecisionTreeClassifier(), 
    ...         n_estimators=50,
    ...         verbose=1).fit(X, y)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])

    """
    def __init__(self, 
            base_estimator=DecisionTreeClassifier(), 
            hardness_func=lambda y_true, y_pred: np.absolute(y_true-y_pred),
            n_estimators=10, 
            k_bins=10, 
            estimator_params = tuple(),
            n_jobs = None,
            random_state = None,
            verbose = 0,):

        self.base_estimator = base_estimator
        self.hardness_func = hardness_func
        self.n_estimators = n_estimators
        self.k_bins = k_bins
        self.estimator_params = estimator_params
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _random_under_sampling(self, X_maj, y_maj, X_min, y_min):
        """Private function used to perform random under-sampling."""

        np.random.seed(self.random_state)
        idx = np.random.choice(len(X_maj), len(X_min), replace=False)
        X_train = np.concatenate([X_maj[idx], X_min])
        y_train = np.concatenate([y_maj[idx], y_min])

        return X_train, y_train

    def _self_paced_under_sampling(self, 
            X_maj, y_maj, X_min, y_min, i_estimator):
        """Private function used to perform self-paced under-sampling."""

        # Update hardness value estimation
        hardness = self.hardness_func(y_maj, self.y_maj_pred_proba_buffer[:, self.class_index_min])

        # If hardness values are not distinguishable, perform random smapling
        if hardness.max() == hardness.min():
            X_train, y_train = self._random_under_sampling(X_maj, y_maj, X_min, y_min)
        # Else allocate majority samples into k hardness bins
        else:
            step = (hardness.max()-hardness.min()) / self.k_bins
            bins = []; ave_contributions = []
            for i_bins in range(self.k_bins):
                idx = (
                    (hardness >= i_bins*step + hardness.min()) & 
                    (hardness < (i_bins+1)*step + hardness.min())
                )
                # Marginal samples with highest hardness value -> kth bin
                if i_bins == (self.k_bins-1):
                    idx = idx | (hardness==hardness.max())
                bins.append(X_maj[idx])
                ave_contributions.append(hardness[idx].mean())

            # Update self-paced factor alpha
            alpha = np.tan(np.pi*0.5*(i_estimator/(self.n_estimators-1)))
            # Caculate sampling weight
            weights = 1 / (ave_contributions + alpha)
            weights[np.isnan(weights)] = 0
            # Caculate sample number from each bin
            n_sample_bins = len(X_min) * weights / weights.sum()
            n_sample_bins = n_sample_bins.astype(int)+1
            
            # Perform self-paced under-sampling
            sampled_bins = []
            for i_bins in range(self.k_bins):
                if min(len(bins[i_bins]), n_sample_bins[i_bins]) > 0:
                    np.random.seed(self.random_state)
                    idx = np.random.choice(
                        len(bins[i_bins]), 
                        min(len(bins[i_bins]), n_sample_bins[i_bins]), 
                        replace=False)
                    sampled_bins.append(bins[i_bins][idx])
            X_train_maj = np.concatenate(sampled_bins, axis=0)
            y_train_maj = np.full(X_train_maj.shape[0], y_maj[0])

            # Handle sparse matrix
            if sp.issparse(X_min):
                X_train = sp.vstack([sp.csr_matrix(X_train_maj), X_min])
            else:
                X_train = np.vstack([X_train_maj, X_min])
            y_train = np.hstack([y_train_maj, y_min])

        return X_train, y_train
        
    def _validate_y(self, y):
        """Validate the label vector."""

        y = column_or_1d(y, warn=True)
        check_classification_targets(y)

        return y
    
    def update_maj_pred_buffer(self, X_maj):
        """Maintain a latest prediction probabilities of the majority 
           training data during ensemble training."""

        if self.n_buffered_estimators_ > len(self.estimators_):
            raise ValueError(
                'Number of buffered estimators ({}) > total estimators ({}), check usage!'.format(
                    self.n_buffered_estimators_, len(self.estimators_)))
        if self.n_buffered_estimators_ == 0:
            self.y_maj_pred_proba_buffer = np.full(shape=(self._n_samples_maj, self.n_classes_), fill_value=1./self.n_classes_)
        y_maj_pred_proba_buffer = self.y_maj_pred_proba_buffer
        for i in range(self.n_buffered_estimators_, len(self.estimators_)):
            y_pred_proba_i = self.estimators_[i].predict_proba(X_maj)
            y_maj_pred_proba_buffer = (y_maj_pred_proba_buffer * i + y_pred_proba_i) / (i+1)
        self.y_maj_pred_proba_buffer = y_maj_pred_proba_buffer
        self.n_buffered_estimators_ = len(self.estimators_)

        return

    def init_data_statistics(self, X, y, label_maj, label_min, to_console=False):
        """Initialize DupleBalance with training data statistics."""

        self._n_samples, self.n_features_ = X.shape
        self.features_ = np.arange(self.n_features_)
        self.org_class_distr = Counter(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_buffered_estimators_ = 0

        if self.n_classes_ != 2:
            raise ValueError(f"Number of classes should be 2, meet {self.n_classes_}, please check usage.")

        if label_maj == None or label_min == None:
            # auto detect majority and minority class label
            sorted_class_distr = sorted(self.org_class_distr.items(), key=lambda d: d[1])
            label_min, label_maj = sorted_class_distr[0][0], sorted_class_distr[1][0]
            if to_console:
                print (f'\n\'label_maj\' and \'label_min\' are not specified, automatically set to {label_maj} and {label_min}')
        
        self.label_maj, self.label_min = label_maj, label_min
        self.class_index_maj, self.class_index_min = list(self.classes_).index(label_maj), list(self.classes_).index(label_min)
        maj_index, min_index = (y==label_maj), (y==label_min)
        self._n_samples_maj, self._n_samples_min = maj_index.sum(), min_index.sum()

        if self._n_samples_maj == 0:
            raise RuntimeWarning(
                f'The specified majority class {self.label_maj} has no data samples, please check usage.')
        if self._n_samples_min == 0:
            raise RuntimeWarning(
                f'The specified minority class {self.label_min} has no data samples, please check usage.')

        self.X_maj, self.y_maj = X[maj_index], y[maj_index]
        self.X_min, self.y_min = X[min_index], y[min_index]
        if to_console:
            print ('----------------------------------------------------')
            print ('# Samples       : {}'.format(self._n_samples))
            print ('# Features      : {}'.format(self.n_features_))
            print ('# Classes       : {}'.format(self.n_classes_))
            cls_label, cls_dis, IRs = '', '', ''
            min_n_samples = min(self.org_class_distr.values())
            for label, num in sorted(self.org_class_distr.items(), key=lambda d: d[1], reverse=True):
                cls_label += f'{label}/'
                cls_dis += f'{num}/'
                IRs += '{:.2f}/'.format(num/min_n_samples)
            print ('Classes         : {}'.format(cls_label[:-1]))
            print ('Class Dist      : {}'.format(cls_dis[:-1]))
            print ('Imbalance Ratio : {}'.format(IRs[:-1]))
            print ('----------------------------------------------------')
            time.sleep(0.25)
        
        return

    def fit(self, X, y, label_maj=None, label_min=None):
        """Build a self-paced ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels).
        
        label_maj : int, optional (default=None)
            The majority class label, default to be `None`.
            if None, `label_maj` will be automatically set when call `fit()`.
            
        label_min : int, optional (default=None)
            The minority class label, default to be `None`.
            if None, `label_min` will be automatically set when call `fit()`.
        
        Returns
        ------
        self : object
        """
        
        # validate data format and estimator
        check_random_state(self.random_state)
        X, y = self._validate_data(
            X, y, accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite=False, multi_output=True)
        y = self._validate_y(y)
        self._validate_estimator()

        # Initialize by spliting majority / minority set
        self.init_data_statistics(
            X, y, label_maj, label_min, 
            to_console=True if self.verbose > 0 else False)
        self.estimators_ = []
        self.estimators_features_ = []

        # Loop start
        if self.verbose > 0:
            iterations = tqdm(range(self.n_estimators))
            iterations.set_description('SPE Training')
        else:
            iterations = range(self.n_estimators)

        for i_iter in iterations:

            # update current majority training data prediction
            self.update_maj_pred_buffer(self.X_maj)
            
            # train a new base estimator and add it into self.estimators_
            X_train, y_train = self._self_paced_under_sampling(
                self.X_maj, self.y_maj, self.X_min, self.y_min, i_iter)
            estimator = self._make_estimator(append=True, random_state=self.random_state)
            estimator.fit(X_train, y_train)
            self.estimators_features_.append(self.features_)

        return self
    
    def _parallel_args(self):
        return {}

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. 
        """

        check_is_fitted(self)
        # Check data
        X = check_array(
            X, accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite=False
        )
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))
        
        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)

        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                             **self._parallel_args())(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X,
                self.n_classes_)
            for i in range(n_jobs))

        # Reduce
        proba = sum(all_proba) / self.n_estimators

        return proba

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)
    
    def score(self, X, y):
        """Returns the average precision score (equivalent to the area under 
        the precision-recall curve) on the given test data and labels.
        
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        score : float
            Average precision of self.predict_proba(X)[:, 1] wrt. y.
        """
        return sklearn.metrics.average_precision_score(
            y, self.predict_proba(X)[:, self.class_index_min])