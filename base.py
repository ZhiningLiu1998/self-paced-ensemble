# -*- coding: utf-8 -*-
"""
@author: ZhiningLiu1998
mailto: znliu19@mails.jlu.edu.cn / zhining.liu@outlook.com
"""

import functools
from functools import update_wrapper
import numpy as np
from joblib import Parallel, effective_n_jobs

import os
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
