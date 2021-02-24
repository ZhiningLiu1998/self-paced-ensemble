"""
--------------------------------------------------------------------------
The `self_paced_ensemble.canonical_resampling` module implement a 
resampling-based classifier for imbalanced classification.
15 resampling algorithms are included: 
'RUS', 'CNN', 'ENN', 'NCR', 'Tomek', 'ALLKNN', 'OSS',
'NM', 'CC', 'SMOTE', 'ADASYN', 'BorderSMOTE', 'SMOTEENN', 
'SMOTETomek', 'ORG'.

Note: the implementation of these resampling algorithms is based on 
imblearn python package. 
See https://github.com/scikit-learn-contrib/imbalanced-learn.
--------------------------------------------------------------------------
"""


from .canonical_resampling import ResampleClassifier

__all__ = [
    "ResampleClassifier",
]
