# -*- coding: utf-8 -*-
"""
A self-paced Ensemble (SPE) Classifier for binary class-imbalanced learning.
    
Self-paced Ensemble (SPE) is an ensemble learning framework for massive highly 
imbalanced classification. It is an easy-to-use solution to class-imbalanced 
problems, features outstanding computing efficiency, good performance, and wide 
compatibility with different learning models.
"""

# Created on Tue May 14 14:32:27 2019
# @author: ZhiningLiu1998
# mailto:  zhining.liu@outlook.com

# %%

from collections import Counter
import numpy as np
from joblib import dump


from .base import BaseImbalancedEnsemble, MAX_INT
from ._self_paced_under_sampler import SelfPacedUnderSampler
from ..utils._validation import _deprecate_positional_args
from ..utils._validation_data import check_eval_datasets
from ..utils._validation_param import (check_target_label_and_n_target_samples, 
                                       check_balancing_schedule, 
                                       check_train_verbose, 
                                       check_eval_metrics,)

# # For local test
# import sys
# sys.path.append("..")
# from self_paced_ensemble.base import BaseImbalancedEnsemble, MAX_INT
# from self_paced_ensemble._self_paced_under_sampler import SelfPacedUnderSampler
# from utils._validation import _deprecate_positional_args
# from utils._validation_data import check_eval_datasets
# from utils._validation_param import (check_target_label_and_n_target_samples, 
#                                      check_balancing_schedule, 
#                                      check_train_verbose, 
#                                      check_eval_metrics,)

# %%

# Properties
_method_name = 'SelfPacedEnsembleClassifier'
_sampler_class = SelfPacedUnderSampler

_solution_type = 'resampling'
_sampling_type = 'under-sampling'
_ensemble_type = 'general'
_training_type = 'iterative'

_properties = {
    'solution_type': _solution_type,
    'sampling_type': _sampling_type,
    'ensemble_type': _ensemble_type,
    'training_type': _training_type,
}

class SelfPacedEnsembleClassifier(BaseImbalancedEnsemble):
    """A self-paced ensemble (SPE) Classifier for class-imbalanced learning.
    
    Self-paced Ensemble (SPE) [1]_ is an ensemble learning framework for massive highly 
    imbalanced classification. It is an easy-to-use solution to class-imbalanced 
    problems, features outstanding computing efficiency, good performance, and wide 
    compatibility with different learning models.

    This implementation extends SPE to support multi-class classification.

    Parameters
    ----------
    base_estimator : estimator object, default=None
        The base estimator to fit on self-paced under-sampled subsets 
        of the dataset. Support for sample weighting is NOT required, 
        but need proper ``classes_`` and ``n_classes_`` attributes. 
        If ``None``, then the base estimator is ``DecisionTreeClassifier()``.

    n_estimators : int, default=50
        The number of base estimators in the ensemble.

    k_bins : int, default=5
        The number of hardness bins that were used to approximate 
        hardness distribution. It is recommended to set it to 5. 
        One can try a larger value when the smallest class in the 
        data set has a sufficient number (say, > 1000) of samples.

    soft_resample_flag : bool, default=False
        Whether to use weighted sampling to perform soft self-paced 
        under-sampling, rather than explicitly cut samples into 
        ``k``-bins and perform hard sampling.

    replacement : bool, default=True
        Whether samples are drawn with replacement. If ``False`` 
        and ``soft_resample_flag = False``, may raise an error when 
        a bin has insufficient number of data samples for resampling.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    {n_jobs}

    {random_state}

    verbose : int, default=0
        Controls the verbosity when predicting.

    Attributes
    ----------
    base_estimator : estimator
        The base estimator from which the ensemble is grown.

    base_sampler_ : SelfPacedUnderSampler
        The base sampler.

    estimators_ : list of estimator
        The collection of fitted base estimators.

    samplers_ : list of SelfPacedUnderSampler
        The collection of fitted samplers.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances if supported by the ``base_estimator``.
    
    estimators_n_training_samples_ : list of ints
        The number of training samples for each fitted 
        base estimators.

    See Also
    --------
    BalanceCascadeClassifier : Ensemble with cascade dynamic under-sampling.

    EasyEnsembleClassifier : Bag of balanced boosted learners.

    RUSBoostClassifier : Random under-sampling integrated in AdaBoost.

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_basic_plot_basic_example.py` for an example.

    References
    ----------
    .. [1] Liu, Z., Cao, W., Gao, Z., Bian, J., Chen, H., Chang, Y., & Liu, T. Y. 
       "Self-paced ensemble for highly imbalanced massive data classification." 
       2020 IEEE 36th International Conference on Data Engineering (ICDE). 
       IEEE, 2010: 841-852.

    Examples
    --------
    {example}
    """
    
    def __init__(self, 
        base_estimator=None, 
        n_estimators:int=50, 
        k_bins:int=5, 
        soft_resample_flag:bool=False, 
        replacement:bool=True, 
        estimator_params=tuple(),
        n_jobs=None,
        random_state=None,
        verbose=0,):
    
        super(SelfPacedEnsembleClassifier, self).__init__( 
            base_estimator=base_estimator, 
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            random_state=random_state,
            n_jobs=n_jobs, 
            verbose=verbose)

        self.__name__ = _method_name
        self.base_sampler = _sampler_class()
        self._sampling_type = _sampling_type
        self._sampler_class = _sampler_class
        self._properties = _properties

        self.k_bins = k_bins
        self.soft_resample_flag = soft_resample_flag
        self.replacement = replacement
    

    def fit(self, X, y, *, sample_weight=None, **kwargs):
        """Build a SPE classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.
        
        target_label : int, default=None
            Specify the class targeted by the under-sampling. 
            All other classes that have more samples than the target class will 
            be considered as majority classes. They will be under-sampled until 
            the number of samples is equalized. The remaining minority classes 
            (if any) will stay unchanged.
        
        n_target_samples : int or dict, default=None
            Specify the desired number of samples (of each class) after the 
            under-sampling. 

            - If ``int``, all classes that have more than the ``n_target_samples`` 
              samples will be under-sampled until the number of samples is equalized.
            - If ``dict``, the keys correspond to the targeted classes. The values 
              correspond to the desired number of samples for each targeted class.
        
        balancing_schedule : str, or callable, default='uniform'
            Scheduler that controls how to sample the data set during the ensemble 
            training process.

            - If ``str``, using the predefined balancing schedule.
              Possible choices are:

                - ``'uniform'``: resample to target distribution for all base estimators;
                - ``'progressive'``: The resample class distributions are progressive 
                  interpolation between the original and the target class distribution.
                  Example: For a class :math:`c`, say the number of samples is :math:`N_{c}` 
                  and the target number of samples is :math:`N'_{c}`. Suppose that we are 
                  training the :math:`t`-th base estimator of a :math:`T`-estimator ensemble, then 
                  we expect to get :math:`(1-\frac{t}{T}) \cdot N_{c} + \frac{t}{T} \cdot N'_{c}` 
                  samples after resampling;

            - If callable, function takes 4 positional arguments with order (``'origin_distr'``: 
              ``dict``, ``'target_distr'``: ``dict``, ``'i_estimator'``: ``int``, ``'total_estimator'``: 
              ``int``), and returns a ``'result_distr'``: ``dict``. For all parameters of type ``dict``, 
              the keys of type ``int`` correspond to the targeted classes, and the values of type ``str`` 
              correspond to the (desired) number of samples for each class.
        
        eval_datasets : dict, default=None
            Dataset(s) used for evaluation during the ensemble training process.
            The keys should be strings corresponding to evaluation datasets' names. 
            The values should be tuples corresponding to the input samples and target
            values. 
            
            Example: ``eval_datasets = {'valid' : (X_valid, y_valid)}``
        
        eval_metrics : dict, default=None
            Metric(s) used for evaluation during the ensemble training process.

            - If ``None``, use 3 default metrics:

                - ``'acc'``: 
                  ``sklearn.metrics.accuracy_score()``
                - ``'balanced_acc'``: 
                  ``sklearn.metrics.balanced_accuracy_score()``
                - ``'weighted_f1'``: 
                  ``sklearn.metrics.f1_score(average='weighted')``

            - If ``dict``, the keys should be strings corresponding to evaluation 
              metrics' names. The values should be tuples corresponding to the metric 
              function (``callable``) and additional kwargs (``dict``).

                - The metric function should at least take 2 named/keyword arguments, 
                  ``y_true`` and one of [``y_pred``, ``y_score``], and returns a float
                  as the evaluation score. Keyword arguments:

                  - ``y_true``, 1d-array of shape (n_samples,), true labels or binary 
                    label indicators corresponds to ground truth (correct) labels.
                  - When using ``y_pred``, input will be 1d-array of shape (n_samples,) 
                    corresponds to predicted labels, as returned by a classifier.
                  - When using ``y_score``, input will be 2d-array of shape (n_samples, 
                    n_classes,) corresponds to probability estimates provided by the 
                    predict_proba method. In addition, the order of the class scores 
                    must correspond to the order of ``labels``, if provided in the metric 
                    function, or else to the numerical or lexicographical order of the 
                    labels in ``y_true``.
                
                - The metric additional kwargs should be a dictionary that specifies 
                  the additional arguments that need to be passed into the metric function. 
            
            Example: ``{'weighted_f1': (sklearn.metrics.f1_score, {'average': 'weighted'})}``
        
        train_verbose : bool, int or dict, default=False
            Controls the verbosity during ensemble training/fitting.

            - If ``bool``: ``False`` means disable training verbose. ``True`` means 
              print training information to sys.stdout use default setting:
              
                - ``'granularity'``        : ``int(n_estimators/10)``
                - ``'print_distribution'`` : ``True``
                - ``'print_metrics'``      : ``True``

            - If ``int``, print information per ``train_verbose`` rounds.

            - If ``dict``, control the detailed training verbose settings. They are:

                - ``'granularity'``: corresponding value should be ``int``, the training
                  information will be printed per ``granularity`` rounds.
                - ``'print_distribution'``: corresponding value should be ``bool``, 
                  whether to print the data class distribution 
                  after resampling. Will be ignored if the 
                  ensemble training does not perform resampling.
                - ``'print_metrics'``: corresponding value should be ``bool``, 
                  whether to print the latest performance score.
                  The performance will be evaluated on the training 
                  data and all given evaluation datasets with the 
                  specified metrics.
              
            .. warning::
                Setting a small ``'granularity'`` value with ``'print_metrics'`` enabled 
                can be costly when the training/evaluation data is large or the metric 
                scores are hard to compute. Normally, one can set ``'granularity'`` to 
                ``n_estimators/10`` (this is used by default).

        Returns
        -------
        self : object
        """
        return super().fit(
            X, y, sample_weight=sample_weight, **kwargs
        )


    @_deprecate_positional_args
    def _fit(self, X, y, 
            *,
            sample_weight=None, 
            target_label:int=None, 
            n_target_samples:int or dict=None, 
            balancing_schedule:str or function='uniform',
            eval_datasets:dict=None,
            eval_metrics:dict=None,
            train_verbose:bool or int or dict=False,
            ):
        
        # X, y, sample_weight, base_estimators_ (default=DecisionTreeClassifier),
        # n_estimators, random_state, sample_weight are already validated in super.fit()
        random_state, n_estimators, replacement, k_bins, soft_resample_flag, classes_ = \
            self.random_state, self.n_estimators, self.replacement, self.k_bins, \
            self.soft_resample_flag, self.classes_

        # Check evaluation data
        check_x_y_args = self.check_x_y_args
        self.eval_datasets_ = check_eval_datasets(eval_datasets, X, y, **check_x_y_args)

        # Check target sample strategy
        origin_distr_ = dict(Counter(y))
        target_label_, target_distr_ = \
            check_target_label_and_n_target_samples(
                y, target_label, n_target_samples, self._sampling_type)
        self.origin_distr_, self.target_label_, self.target_distr_ = \
            origin_distr_, target_label_, target_distr_
        
        # Check balancing schedule
        balancing_schedule_ = check_balancing_schedule(balancing_schedule)
        self.balancing_schedule_ = balancing_schedule_

        # Check evaluation metrics
        self.eval_metrics_ = check_eval_metrics(eval_metrics)
        
        # Check training train_verbose format
        self.train_verbose_ = check_train_verbose(
            train_verbose, self.n_estimators, **self._properties)
        
        # Set training verbose format
        self._init_training_log_format()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimators_features_ = []
        self.estimators_n_training_samples_ = np.zeros(n_estimators, dtype=np.int)
        self.samplers_ = []
        
        # Genrate random seeds array
        seeds = random_state.randint(MAX_INT, size=n_estimators)
        self._seeds = seeds

        # Check if sample_weight is specified
        specified_sample_weight = (sample_weight is not None)

        for i_iter in range(n_estimators):

            current_iter_distr = balancing_schedule_(
                origin_distr=origin_distr_,
                target_distr=target_distr_,
                i_estimator=i_iter,
                total_estimator=n_estimators,
            )

            sampler = self._make_sampler(
                append=True,
                random_state=seeds[i_iter],
                sampling_strategy=current_iter_distr,
                k_bins=k_bins,
                soft_resample_flag=soft_resample_flag, 
                replacement=replacement, 
            )
            
            # update self.y_pred_proba_latest
            self._update_cached_prediction_probabilities(i_iter, X)
            
            # compute alpha
            alpha = np.tan(np.pi*0.5*(i_iter/(max(n_estimators-1, 1))))

            # Perform self-paced under-sampling
            resample_out = sampler.fit_resample(X, y, 
                    y_pred_proba=self.y_pred_proba_latest,
                    alpha=alpha,
                    classes_=classes_,
                    sample_weight=sample_weight)

            # Train a new base estimator on resampled data 
            # and add it into self.estimators_
            estimator = self._make_estimator(append=True, random_state=seeds[i_iter])
            if specified_sample_weight:
                (X_resampled, y_resampled, sample_weight_resampled) = resample_out
                estimator.fit(X_resampled, y_resampled, sample_weight=sample_weight_resampled)
            else: 
                (X_resampled, y_resampled) = resample_out
                estimator.fit(X_resampled, y_resampled)

            self.estimators_features_.append(self.features_)
            self.estimators_n_training_samples_[i_iter] = y_resampled.shape[0]

            # Print training infomation to console.
            self._training_log_to_console(i_iter, y_resampled)
        
        return self

    
    def _update_cached_prediction_probabilities(self, i_iter, X):
        """Private function that maintains a latest prediction probabilities of the training
         data during ensemble training. Must be called in each iteration before fit the 
         base_estimator."""

        if i_iter == 0:
            self.y_pred_proba_latest = np.zeros((self._n_samples, self.n_classes_), 
                                                dtype=np.float64)
        else:
            y_pred_proba_latest = self.y_pred_proba_latest
            y_pred_proba_new = self.estimators_[-1].predict_proba(X)
            self.y_pred_proba_latest = (y_pred_proba_latest * i_iter + y_pred_proba_new) / (i_iter+1)
        return
    
    def save(self, filename:str):
        """Save the model to a file."""
        dump(self, filename=filename)

# %%

if __name__ == '__main__':
    from collections import Counter
    from copy import copy
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    
    # X, y = make_classification(n_classes=2, class_sep=2, # 2-class
    #     weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    #     n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    X, y = make_classification(n_classes=3, class_sep=2, # 3-class
        weights=[0.1, 0.3, 0.6], n_informative=3, n_redundant=1, flip_y=0,
        n_features=20, n_clusters_per_class=1, n_samples=2000, random_state=10)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=42)

    origin_distr = dict(Counter(y_train)) # {2: 600, 1: 300, 0: 100}
    print('Original training dataset shape %s' % origin_distr)

    target_distr = {2: 200, 1: 100, 0: 100}

    init_kwargs_default = {
        'base_estimator': None,
        # 'base_estimator': DecisionTreeClassifier(max_depth=10),
        'n_estimators': 100,
        'k_bins': 5,
        'soft_resample_flag': False,
        'replacement': True,
        'estimator_params': tuple(),
        'n_jobs': None,
        'random_state': 42,
        # 'random_state': None,
        'verbose': 0,
    }

    fit_kwargs_default = {
        'X': X_train,
        'y': y_train,
        'sample_weight': None,
        'target_label': None,
        'n_target_samples': None,
        # 'n_target_samples': target_distr,
        'balancing_schedule': 'uniform',
        'eval_datasets': {'valid': (X_valid, y_valid)},
        'eval_metrics': {
            'acc': (accuracy_score, {}),
            'balanced_acc': (balanced_accuracy_score, {}),
            'weighted_f1': (f1_score, {'average':'weighted'}),},
        'train_verbose': {
            'granularity': 10,
            'print_distribution': True,
            'print_metrics': True,},
    }

    ensembles = {}

    init_kwargs, fit_kwargs = copy(init_kwargs_default), copy(fit_kwargs_default)
    spe = SelfPacedEnsembleClassifier(**init_kwargs).fit(**fit_kwargs)
    ensembles['spe'] = spe

    init_kwargs, fit_kwargs = copy(init_kwargs_default), copy(fit_kwargs_default)
    fit_kwargs.update({
        'balancing_schedule': 'progressive'
    })
    spe_prog = SelfPacedEnsembleClassifier(**init_kwargs).fit(**fit_kwargs)
    ensembles['spe_prog'] = spe_prog

    init_kwargs, fit_kwargs = copy(init_kwargs_default), copy(fit_kwargs_default)
    init_kwargs.update({
        'soft_resample_flag': True,
        'replacement': False,
    })
    spe_soft = SelfPacedEnsembleClassifier(**init_kwargs).fit(**fit_kwargs)
    ensembles['spe_soft'] = spe_soft


    # %%
    from imbalanced_ensemble.visualizer import ImbalancedEnsembleVisualizer

    visualizer = ImbalancedEnsembleVisualizer(
        eval_datasets = None,
        eval_metrics = None,
    ).fit(
        ensembles = ensembles,
        granularity = 5,
    )
    fig, axes = visualizer.performance_lineplot(
        on_ensembles=None,
        on_datasets=None,
        split_by=[],
        n_samples_as_x_axis=False,
        sub_figsize=(4, 3.3),
        sup_title=True,
        alpha=0.8,
    )
    fig, axes = visualizer.confusion_matrix_heatmap(
        on_ensembles=None,
        on_datasets=None,
        sub_figsize=(4, 3.3),
    )

    # %%
