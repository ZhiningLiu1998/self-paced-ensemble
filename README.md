# Self-paced Ensmeble

**Self-paced Ensemble (SPE) is a learning framework for massive highly imbalanced classification.**  
1. It utilizes under-sampling without computing distance between data instances, thus is computational efficient even when applying on the large-scale dataset. 
2. It can be used to boost any canonical classifier's performance on highly imbalanced datasets as long as the classifier provides classification probability.
3. It demonstrates superior performance on various real-world tasks while being robust to noises and missing values.
4. It is applicable to most datasets since it does not require any pre-defined distance metric.

This repository contains:

## 1. Additional experimental results:

### Results on additional datasets
**We introduce 7 additional public datasets to validate our method SPE.**  

Their properties vary widely from one another, with IR ranging from 9.1:1 to 111:1, dataset sizes ranging from 360 to 145,751, and feature numbers ranging from 6 to 617. See the table below for more information about these datasets.

![image](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/pic/additional_datasets.png)

We compare our methods SPE with other 5 ensemble-based imbalance learning methods:  
Over-sampling ensemble:  **SMOTEBoost, SMOTEBagging**  
Under-sampling ensemble: **RUSBoost, UnderBagging, BalanceCascade**  

We use Decision Tree as the base classifier for all ensemble methods as other classifiers such as KNN do not support Boosting-based methods. We implemented SPE with Absolute Error as the hardness function and set k=10. In each dataset, 80% samples were used for training. The rest 20% was used as the test set. All the experimental results were reported on the test set (mean and standard deviation of 50 independent runs with different random seeds for training base classifiers). 

![image](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/pic/additional_datasets_results.png)

### Results using additional classifiers (supplementary of Table IV)

![image](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/pic/statistics.png)

**Dataset Links used in Table IV**  
Credit Fraud: https://www.kaggle.com/mlg-ulb/creditcardfraud  
KDDCUP: https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data  
Record Linkage: https://archive.ics.uci.edu/ml/datasets/Record+Linkage+Comparison+Patterns  
Payment Simulation: https://www.kaggle.com/ntnu-testimon/paysim1  

![image](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/pic/credit.png)
![image](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/pic/kddcup.png)
![image](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/pic/record_paysim.png)

## 2. Implementation of Self-paced Ensemble

Preparing for release... XD
