# -*- coding: utf-8 -*-
"""
Five ensemble learning algorithms for imbalanced classification, including:
'SMOTEBaggingClassifier', 'SMOTEBoostClassifier', 'RUSBoostClassifier', 
'UnderBaggingClassifier', and 'BalanceCascadeClassifier'.

Note: methods in this module are now included in the `imbalanced-ensemble`.
Please refer to https://imbalanced-ensemble.readthedocs.io/ for more details.
"""

# Created on Sun Jan 13 14:32:27 2019
# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%

from imbens.ensemble import SMOTEBoostClassifier
from imbens.ensemble import SMOTEBaggingClassifier
from imbens.ensemble import RUSBoostClassifier
from imbens.ensemble import UnderBaggingClassifier
from imbens.ensemble import BalanceCascadeClassifier