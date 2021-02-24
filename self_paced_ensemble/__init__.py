"""
``self-paced ensemble`` is a python-based ensemble learning framework for 
dealing with binary class-imbalanced classification problems in machine learning.

Subpackages
-----------
self_paced_ensemble
    Module which provides our SelfPacedEnsembleClassifier implementation.
canonical_ensemble
    Module which provides baseline methods based on ensemble learning.
canonical_resampling
    Module which provides baseline methods based on data resampling.
utils
    Module including various utilities.
"""

from . import self_paced_ensemble
from . import canonical_ensemble
from . import canonical_resampling
from . import utils

from .self_paced_ensemble import SelfPacedEnsembleClassifier

from .__version__ import __version__

__all__ = [
    "SelfPacedEnsembleClassifier",
    "self_paced_ensemble",
    "canonical_ensemble",
    "canonical_resampling",
    "utils",
    "__version__",
]