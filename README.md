<!-- <h1 align="center"> Self-paced Ensemble </h1> -->

![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/spe/spe_header.png)

<p align="center">
  <a href="https://github.com/ZhiningLiu1998/self-paced-ensemble">
    <img src="https://img.shields.io/badge/ZhiningLiu1998-SPE-orange">
  </a>
  <a href="https://github.com/ZhiningLiu1998/self-paced-ensemble/stargazers">
    <img src="https://img.shields.io/github/stars/ZhiningLiu1998/self-paced-ensemble">
  </a>
  <a href="https://github.com/ZhiningLiu1998/self-paced-ensemble/network/members">
    <img src="https://img.shields.io/github/forks/ZhiningLiu1998/self-paced-ensemble">
  </a>
  <a href="https://github.com/ZhiningLiu1998/self-paced-ensemble/issues">
    <img src="https://img.shields.io/github/issues/ZhiningLiu1998/self-paced-ensemble">
  </a>
  <a href="https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/ZhiningLiu1998/self-paced-ensemble">
  </a>
  <a href="https://pypi.org/project/self-paced-ensemble/">
    <img src="https://badge.fury.io/py/self-paced-ensemble.svg">
  </a>
  <br>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/pypi/pyversions/self-paced-ensemble.svg">
  </a>
  <a href="https://github.com/ZhiningLiu1998/self-paced-ensemble/graphs/traffic">
    <img src="https://visitor-badge.glitch.me/badge?page_id=ZhiningLiu1998.self-paced-ensemble">
  </a>
  <a href="https://pepy.tech/project/self-paced-ensemble">
    <img src="https://pepy.tech/badge/self-paced-ensemble">
  </a>
  <a href="https://pepy.tech/project/self-paced-ensemble">
    <img src="https://pepy.tech/badge/self-paced-ensemble/month">
  </a>
</p>


<h3 align="center"> "Self-paced Ensemble for Highly Imbalanced Massive Data Classification"
<br>
(ICDE 2020)
[<a href="http://zhiningliu.com/files/ICDE_2020_self_paced_ensemble.pdf">Paper</a>]
[<a href="http://zhiningliu.com/files/ICDE_2020_self_paced_ensemble_slides.pdf">Slides</a>]
[<a href="https://arxiv.org/abs/1909.03500v3">arXiv</a>]
[<a href="https://pypi.org/project/self-paced-ensemble">PyPI</a>]
[<a href="https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.SelfPacedEnsembleClassifier.html">Documentation</a>]
</h3>


**Self-paced Ensemble (SPE) is an ensemble learning framework for massive highly imbalanced classification. It is an easy-to-use solution to class-imbalanced problems, features outstanding computing efficiency, good performance, and wide compatibility with different learning models. This SPE implementation supports multi-class classification.**

<table><tr><td bgcolor=MistyRose align="center"><strong>
<font color='red'>Note: </font> 
<font color=Navy> 
SPE is now a part of <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble"> imbalanced-ensemble </a> [<a href="https://imbalanced-ensemble.readthedocs.io/en/latest/">Doc</a>, <a href="https://pypi.org/project/imbalanced-ensemble/">PyPI</a>]. Try it for more methods and advanced features!
</font>
</strong></td></tr></table>

# Cite Us

**If you find this repository helpful in your work or research, we would greatly appreciate citations to the following [paper](https://arxiv.org/pdf/1909.03500v3.pdf):**

```bib
@inproceedings{liu2020self-paced-ensemble,
    title={Self-paced Ensemble for Highly Imbalanced Massive Data Classification},
    author={Liu, Zhining and Cao, Wei and Gao, Zhifeng and Bian, Jiang and Chen, Hechang and Chang, Yi and Liu, Tie-Yan},
    booktitle={2020 IEEE 36th International Conference on Data Engineering (ICDE)},
    pages={841--852},
    year={2020},
    organization={IEEE}
}
```

# Install

It is recommended to use **pip** for installation.  
Please make sure the **latest version** is installed to avoid potential problems:
```shell
$ pip install self-paced-ensemble            # normal install
$ pip install --upgrade self-paced-ensemble  # update if needed
```

Or you can install SPE by clone this repository:
```shell
$ git clone https://github.com/ZhiningLiu1998/self-paced-ensemble.git
$ cd self-paced-ensemble
$ python setup.py install
```

Following dependencies are required:
- [python](https://www.python.org/) (>=3.6)
- [numpy](https://numpy.org/) (>=1.13.3)
- [scipy](https://www.scipy.org/) (>=0.19.1)
- [joblib](https://pypi.org/project/joblib/) (>=0.11)
- [scikit-learn](https://scikit-learn.org/stable/) (>=0.24)
- [imblearn](https://pypi.org/project/imblearn/) (>=0.7.0)
- [imbalanced-ensemble](https://pypi.org/project/imbalanced-ensemble/) (>=0.1.3)

# Table of Contents

- [Cite Us](#cite-us)
- [Install](#install)
- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Documentation](#documentation)
- [Examples](#examples)
- [Results](#results)
- [Miscellaneous](#miscellaneous)
- [References](#references)

# Background

SPE performs strictly balanced under-sampling in each iteration and is therefore very *computationally efficient*. In addition, SPE does not rely on calculating the distance between samples to perform resampling. It can be easily applied to datasets that lack well-defined distance metrics (e.g. with categorical features / missing values) without any modification. Moreover, as a *generic ensemble framework*, our methods can be easily adapted to most of the existing learning methods (e.g., C4.5, SVM, GBDT, and Neural Network) to boost their performance on imbalanced data. Compared to existing imbalance learning methods, *SPE works particularly well on datasets that are large-scale, noisy, and highly imbalanced (e.g. with imbalance ratio greater than 100:1).* Such kind of data widely exists in real-world industrial applications. The figure below gives an overview of the SPE framework.

![image](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/spe/framework.png)

# Documentation

**Our SPE implementation can be used much in the same way as the [`sklearn.ensemble`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble) classifiers. Detailed documentation of ``SelfPacedEnsembleClassifier`` can be found [HERE](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.SelfPacedEnsembleClassifier.html).**

# Examples

You can check out [**examples using SPE**](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.SelfPacedEnsembleClassifier.html#examples-using-imbalanced-ensemble-ensemble-selfpacedensembleclassifier) for more comprehensive usage examples.


![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/example_gallery_snapshot.png)


**API demo**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Prepare class-imbalanced train & test data
X, y = make_classification(n_classes=2, random_state=42, weights=[0.1, 0.9])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)

# Train an SPE classifier
clf = SelfPacedEnsembleClassifier(
        base_estimator=DecisionTreeClassifier(), 
        n_estimators=10,
    ).fit(X_train, y_train)

# Predict with an SPE classifier
clf.predict(X_test)
```

**Advanced usage example**

Please see [usage_example.ipynb](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/examples/usage_example.ipynb).

**Compare SPE with other methods**

Please see [comparison_example.ipynb](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/examples/comparison_example.ipynb).

# Results

Dataset links:
[Credit Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud), 
[KDDCUP](https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data), 
[Record Linkage](https://archive.ics.uci.edu/ml/datasets/Record+Linkage+Comparison+Patterns), 
[Payment Simulation](https://www.kaggle.com/ealaxi/paysim1).  

![image](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/spe/statistics.png)  

Comparisons of SPE with traditional resampling/ensemble methods in terms of performance & computational efficiency.

<!-- ![image](https://github.com/ZhiningLiu1998/figures/blob/master/spe/results.png) -->

![image](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/spe/results_resampling.png)

![image](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/spe/results_ensemble.png)

![image](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/spe/results_ensemble_curve.png)

# Miscellaneous

**This repository contains:**
- Implementation of Self-paced Ensemble
- Implementation of 5 ensemble-based imbalance learning baselines
  - `SMOTEBoost` [1]
  - `SMOTEBagging` [2]
  - `RUSBoost` [3]
  - `UnderBagging` [4]
  - `BalanceCascade` [5]
- Implementation of resampling based imbalance learning baselines [6]
- Additional experimental results

**NOTE:** The implementations of other ensemble and resampling methods are based on [imbalanced-ensemble](https://github.com/ZhiningLiu1998/imbalanced-ensemble) and [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn).

Check out our other works:
- [IMBENS: class-imbalanced ensemble learning in Python](https://github.com/ZhiningLiu1998/imbalanced-ensemble) (Python package)
- [Awesome-imbalanced-learning: paper, code, frameworks, and libraries](https://github.com/ZhiningLiu1998/awesome-imbalanced-learning) (Resource list)
- [MESA: Boost Ensemble Imbalanced Learning with MEta-SAmpler](https://github.com/ZhiningLiu1998/mesa) (Meta-imbalanced-learning)

# References

| #   | Reference |
|-----|-------|
| [1] | N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer, Smoteboost: Improving prediction of the minority class in boosting. in European conference on principles of data mining and knowledge discovery. Springer, 2003, pp. 107–119|
| [2] | S. Wang and X. Yao, Diversity analysis on imbalanced data sets by using ensemble models. in 2009 IEEE Symposium on Computational Intelligence and Data Mining. IEEE, 2009, pp. 324–331.|
| [3] | C. Seiffert, T. M. Khoshgoftaar, J. Van Hulse, and A. Napolitano, “Rusboost: A hybrid approach to alleviating class imbalance,” IEEE Transactions on Systems, Man, and Cybernetics-Part A: Systems and Humans, vol. 40, no. 1, pp. 185–197, 2010.|
| [4] | R. Barandela, R. M. Valdovinos, and J. S. Sanchez, “New applications´ of ensembles of classifiers,” Pattern Analysis & Applications, vol. 6, no. 3, pp. 245–256, 2003.|
| [5] | X.-Y. Liu, J. Wu, and Z.-H. Zhou, “Exploratory undersampling for class-imbalance learning,” IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539–550, 2009.|
| [6] | Guillaume Lemaître, Fernando Nogueira, and Christos K. Aridas. Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning. Journal of Machine Learning Research, 18(17):1–5, 2017.|