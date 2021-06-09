"""
--------------------------------------------------------------------------
The `self_paced_ensemble.self_paced_ensemble` module implement a 
self-paced Ensemble (SPE) Classifier for binary class-imbalanced learning.
    
Self-paced Ensemble (SPE) is an ensemble learning framework for massive highly 
imbalanced classification. It is an easy-to-use solution to class-imbalanced 
problems, features outstanding computing efficiency, good performance, and wide 
compatibility with different learning models.

See https://github.com/ZhiningLiu1998/self-paced-ensemble.

Reference:
Liu Z, Cao W, Gao Z, et al. Self-paced ensemble for highly imbalanced 
massive data classification[C]//2020 IEEE 36th International Conference 
on Data Engineering (ICDE). IEEE, 2020: 841-852.
--------------------------------------------------------------------------
"""

from ._self_paced_ensemble import SelfPacedEnsembleClassifier

__all__ = [
    "SelfPacedEnsembleClassifier",
]
