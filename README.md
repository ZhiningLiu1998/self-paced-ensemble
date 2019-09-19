# Self-paced Ensmeble

**Self-paced Ensemble (SPE) is a general learning framework for massive highly imbalanced classification.**

SPE performs strictly balanced under-sampling in each iteration and is therefore very computationally efficient. In addition, SPE does not rely on calculating the distance between samples to perform resampling. It can be easily applied to datasets that lack well-defined distance metrics (e.g. with categorical features / missing values) without any modification. Compared to existing imbalance learning methods, SPE works particularly well on datasets that are large-scale, noisy, and highly imbalanced (e.g. with imbalance ratio greater than 100:1). Such kind of data widely exists in real-world industrial applications.

**This repository contains:**
- Implementation of Self-paced Ensemble
- Implementation of 5 ensemble-based imbalance learning baselines
  - `SMOTEBoost` [1]
  - `SMOTEBagging` [2]
  - `RUSBoost` [3]
  - `UnderBagging` [4]
  - `BalanceCascade` [5]
- Implementation of 15 resampling based imbalance learning baselines
- Additional experimental results

**NOTE:** The implementations of [1],[3] and resampling methods are based on [imbalanced-algorithms](https://github.com/dialnd/imbalanced-algorithms) and [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn).

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
  - [Documentation](#documentation)
  - [Examples](#examples)
  - [Conducting comparative experiments](#conducting-comparative-experiments)
- [Additional experimental results](#additional-experimental-results)
  - [Results on additional datasets](#results-on-additional-datasets)
  - [Results using additional classifiers](#results-using-additional-classifiers)


## Background

The rising big data era has been witnessing more classification tasks with largescale but extremely imbalance and low-quality datasets. Most of existing learning methods suffer from poor performance or low computation efficiency under such a scenario. To tackle this problem, we conduct deep investigations into the nature of class imbalance, which reveals that not only the disproportion between classes, but also other difficulties embedded in the nature of data, especially, noises and class overlapping, prevent us from learning effective classifiers. Taking those factors into consideration, we propose a novel framework for imbalance classification that aims to generate a strong ensemble by self-paced harmonizing data hardness via under-sampling. Extensive experiments have shown that this new framework, while being very computationally efficient, can lead to robust performance even under highly overlapping classes and extremely skewed distribution. Note that, our methods can be easily adapted to most of existing learning methods (e.g., C4.5, SVM, GBDT and Neural Network) to boost their performance on imbalanced data. The figure below gives an overview of the SPE framework.

![image](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/pic/framework.png)

## Install

Our SPE implementation requires following dependencies:
- [python](https://www.python.org/) (>=3.5)
- [numpy](https://numpy.org/) (>=1.11)
- [scipy](https://www.scipy.org/) (>=0.17)
- [scikit-learn](https://scikit-learn.org/stable/) (>=0.21)

Currently you can install SPE by clone this repository. We'll release SPE on the PyPI in the future.

```
git clone https://github.com/ZhiningLiu1998/self-paced-ensemble.git
```

## Usage

### Documentation

**Our SPE implementation can be used much in the same way as the ensemble classifiers in [sklearn.ensemble](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble).**  

| Parameters    | Description   |
| ------------- | ------------- |
| `base_estimator` | *object, optional (default=`sklearn.tree.DecisionTreeClassifier()`)* <br> The base estimator to fit on self-paced under-sampled subsets of the dataset. NO need to support sample weighting. Built-in `fit()`, `predict()`, `predict_proba()` methods are required. |
| `hardness_func`  | *function, optional (default=`lambda y_true, y_pred: np.absolute(y_true-y_pred)`)* <br> User-specified classification hardness function. <br> Input: `y_true` and `y_pred` Output: `hardness` (1-d array)  |
| `n_estimator`    | *integer, optional (default=10)* <br> The number of base estimators in the ensemble. |
| `k_bins`         | *integer, optional (default=10)* <br> The number of hardness bins that were used to approximate hardness distribution. |
| `random_state`   | *integer / RandomState instance / None, optional (default=None)* <br> If integer, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `numpy.random`. |

----------------

| Methods    | Description   |
| ---------- | ------------- |
| `fit(self, X, y, label_maj=0, label_min=1)` | Build a self-paced ensemble of estimators from the training set (X, y). <br> `label_maj`/`label_min` specify the label of majority/minority class. <br> By default, we let the minority class be positive class (`label_min=1`). |
| `predict(self, X)` | Predict class for X. |
| `predict_proba(self, X)` | Predict class probabilities for X. |
| `predict_log_proba(self, X)` | Predict class log-probabilities for X. |
| `score(self, X, y)` | Returns the average precision score on the given test data and labels. |

----------------

| Attributes    | Description   |
| ------------- | ------------- |
| `base_estimator_` | *estimator* <br> The base estimator from which the ensemble is grown. |
| `estimators_` | *list of estimator* <br> The collection of fitted base estimators. |

### Examples

**A minimal example**
```
X, y = <data_loader>.load_data()
spe = SelfPacedEnsemble().fit(X, y)
```

**A non-minimal working example** (It demonstrates some of the features of SPE)
```
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from src.self_paced_ensemble import SelfPacedEnsemble
from src.utils import (make_binary_classification_target, imbalance_train_test_split)

X, y = datasets.fetch_covtype(return_X_y=True)
y = make_binary_classification_target(y, pos_label=7, verbose=True)
X_train, X_test, y_train, y_test = imbalance_train_test_split(X, y, test_size=0.2)

def absolute_error(y_true, y_pred):
    """Self-defined classification hardness function"""
    return np.absolute(y_true - y_pred)

spe = SelfPacedEnsemble(
    base_estimator=DecisionTreeClassifier(),
    hardness_func=absolute_error,
    n_estimators=10,
    ).fit(X_train, y_train)

print('auc_prc_score: {}'.format(spe.score(X_test, y_test)))
```

### Conducting comparative experiments

We also provide a simple framework ([*run_example.py*](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/run_example.py)) for conveniently comparing the performance of our method and other baselines. It is also a more complex example of how to use our implementation of ensemble methods to perform classification. To use it, simply run:

```
python run_example.py --method=SPEnsemble --n-estimators=10 --runs=10
```
You should expect output console log like this:
```
Running method:         SPEnsemble - 10 estimators in 10 independent run(s) ...
100%|█████████████████████████████████████████| 10/10 [00:14<00:00,  1.42s/it]]
ave_run_time:           0.686s
------------------------------
Metrics:
AUCPRC  mean:0.910  std:0.009
F1      mean:0.872  std:0.006
G-mean  mean:0.873  std:0.007
MCC     mean:0.868  std:0.007
```

| Arguments   | Description   |
| ----------- | ------------- |
| `--method` | *string, optional (default=`'SPEnsemble'`)* <br> support: `SPEnsemble`, `SMOTEBoost`, `SMOTEBagging`, `RUSBoost`, `UnderBagging`, `Cascade`, `all` <br> When `all`, the script will run all supported methods. |
| `--n-estimators` | *integer, optional (default=10)* <br> The number of base estimators in the ensemble. |
| `--runs` | *integer, optional (default=10)* <br> The number of independent runs for evaluating method performance. |


## Additional experimental results

### Results on additional datasets
**We introduce 7 additional public datasets to validate our method SPE.**  

Their properties vary widely from one another, with IR ranging from 9.1:1 to 111:1, dataset sizes ranging from 360 to 145,751, and feature numbers ranging from 6 to 617. See the table below for more information about these datasets.

![image](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/pic/additional_datasets.png)

SPE was compared with other 5 ensemble-based imbalance learning methods:  
Over-sampling-based ensemble:  `SMOTEBoost`, `SMOTEBagging`  
Under-sampling-based ensemble: `RUSBoost`, `UnderBagging`, `Cascade`  
We use Decision Tree as the base classifier for all ensemble methods as other classifiers such as KNN do not support Boosting-based methods. We implemented SPE with Absolute Error as the hardness function and set k=10. In each dataset, 80% samples were used for training. The rest 20% was used as the test set. All the experimental results were reported on the test set (mean and standard deviation of 50 independent runs with different random seeds for training base classifiers). 

![image](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/pic/additional_datasets_results.png)

### Results using additional classifiers

**(supplementary of Table IV)**  
Dataset links:
[Credit Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud), 
[KDDCUP](https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data), 
[Record Linkage](https://archive.ics.uci.edu/ml/datasets/Record+Linkage+Comparison+Patterns), 
[Payment Simulation](https://www.kaggle.com/ntnu-testimon/paysim1).

![image](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/pic/statistics.png)  
------
![image](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/pic/credit.png)
![image](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/pic/kddcup.png)
![image](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/pic/record_paysim.png)

## References

> [1] N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer, “Smoteboost: Improving prediction of the minority class in boosting,” in European conference on principles of data mining and knowledge discovery. Springer, 2003, pp. 107–119  
> [2] S. Wang and X. Yao, “Diversity analysis on imbalanced data sets by using ensemble models,” in 2009 IEEE Symposium on Computational Intelligence and Data Mining. IEEE, 2009, pp. 324–331.  
> [3] C. Seiffert, T. M. Khoshgoftaar, J. Van Hulse, and A. Napolitano, “Rusboost: A hybrid approach to alleviating class imbalance,” IEEE Transactions on Systems, Man, and Cybernetics-Part A: Systems and Humans, vol. 40, no. 1, pp. 185–197, 2010.  
> [4] R. Barandela, R. M. Valdovinos, and J. S. Sanchez, “New applications´ of ensembles of classifiers,” Pattern Analysis & Applications, vol. 6, no. 3, pp. 245–256, 2003.  
> [5] X.-Y. Liu, J. Wu, and Z.-H. Zhou, “Exploratory undersampling for class-imbalance learning,” IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539–550, 2009.  
