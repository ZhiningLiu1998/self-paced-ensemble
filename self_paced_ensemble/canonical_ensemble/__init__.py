"""
------------------------------------------------------------------------------
The `self_paced_ensemble.canonical_ensemble` module implement 5 ensemble 
learning algorithms for imbalanced classification, including:
'SMOTEBaggingClassifier', 'SMOTEBoostClassifier', 'RUSBoostClassifier', 
'UnderBaggingClassifier', and 'BalanceCascadeClassifier'.

Note: methods in this module are now included in the `imbalanced-ensemble`.
Please refer to https://imbalanced-ensemble.readthedocs.io/ for more details.
------------------------------------------------------------------------------
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
