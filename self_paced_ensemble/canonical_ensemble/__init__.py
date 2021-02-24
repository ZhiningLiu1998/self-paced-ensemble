"""
--------------------------------------------------------------------------
The `self_paced_ensemble.canonical_ensemble` module implement 5 ensemble 
learning algorithms for imbalanced classification, including:
'SMOTEBaggingClassifier', 'SMOTEBoostClassifier', 'RUSBoostClassifier', 
'UnderBaggingClassifier', and 'BalanceCascadeClassifier'.

Note: the implementation of SMOTEBoost&RUSBoost was obtained from
imbalanced-algorithms. See https://github.com/dialnd/imbalanced-algorithms.
--------------------------------------------------------------------------
"""

from .canonical_ensemble import (
    SMOTEBaggingClassifier,
    SMOTEBoostClassifier,
    RUSBoostClassifier,
    UnderBaggingClassifier,
    BalanceCascadeClassifier,
)

__all__ = [
    "SMOTEBaggingClassifier",
    "SMOTEBoostClassifier",
    "RUSBoostClassifier",
    "UnderBaggingClassifier",
    "BalanceCascadeClassifier",
]
