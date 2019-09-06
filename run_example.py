"""
In this python script we provided an example of how to use our 
implementation of ensemble methods to perform classification.

Usage:
```
python run_example.py --method=SPEnsemble --n_estimators=10 --runs=10
```
or with shortopts:
```
python run_example.py -m SPEnsemble -n 10 -r 10
```

run arguments:
    -m / --methods: string
    |   Specify which method were used to build the ensemble classifier.
    |   support: 'SPEnsemble', 'SMOTEBoost', 'SMOTEBagging', 'RUSBoost', 'UnderBagging', 'Cascade'
    -n / --n_estimators: integer
    |   Specify how much base estimators were used in the ensemble.
    -r / --runs: integer
    |   Specify the number of independent runs (to obtain mean and std)

"""

from time import clock
import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
import sys, getopt
import warnings
warnings.filterwarnings("ignore")

from src.self_paced_ensemble import SelfPacedEnsemble
from src.canonical_ensemble import (
    SMOTEBagging, SMOTEBoost, UnderBagging, RUSBoost, BalanceCascade
)
from src.utils import (
    auc_prc, f1_optim, gm_optim, mcc_optim, parse, Error,
    imbalance_train_test_split, imbalance_random_subset,
    make_binary_classification_target,
)

METHODS = ['SPEnsemble', 'SMOTEBoost', 'SMOTEBagging', 'RUSBoost', 'UnderBagging', 'Cascade']
RANDOM_STATE = 42

def init_model(method, base_estimator, n_estimators):
    '''
    return a model specified by "method".
    '''
    if method == 'SPEnsemble':
        model = SelfPacedEnsemble(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'SMOTEBoost':
        model = SMOTEBoost(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'SMOTEBagging':
        model = SMOTEBagging(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'RUSBoost':
        model = RUSBoost(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'UnderBagging':
        model = UnderBagging(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'Cascade':
        model = BalanceCascade(base_estimator = base_estimator, n_estimators = n_estimators)
    else:
        raise Error('No such method support: {}'.format(method))
    return model

def main():
    # Parse arguments
    method_used, n_estimators, runs = parse(sys.argv, supported_methods=METHODS+['all'])

    # Load & Split training/test data
    print ('\nDataset used: \t\tForest covertypes from UCI (10% random subset)')
    X, y = datasets.fetch_covtype(return_X_y=True)
    y = make_binary_classification_target(y, 7, verbose=True)
    X, y = imbalance_random_subset(
        X, y, size=0.1, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = imbalance_train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Train & Record
    method_list = METHODS if method_used=='all' else [method_used]
    for method in method_list:
        print('\nRunning ...')
        print('method_name:\t\t{}'.format(method))
        print('n_estimators:\t\t{}'.format(n_estimators))
        print('independent_run(s):\t{}'.format(runs))
        scores = []; times = []
        for _ in range(runs):
            model = init_model(
                method=method,
                n_estimators=n_estimators,
                base_estimator=sklearn.tree.DecisionTreeClassifier(),
            )
            start_time = clock()
            model.fit(X_train, y_train)
            times.append(clock()-start_time)
            y_pred = model.predict_proba(X_test)[:, 1]
            scores.append([
                auc_prc(y_test, y_pred),
                f1_optim(y_test, y_pred),
                gm_optim(y_test, y_pred),
                mcc_optim(y_test, y_pred)
            ])
        
        # Print results to console
        print('ave_run_time:\t\t{:.3f}s'.format(np.mean(times)))
        print('--------------------------')
        print('Metrics:')
        df_scores = pd.DataFrame(scores, columns=['AUCPRC', 'F1', 'G-mean', 'MCC'])
        for metric in df_scores.columns.tolist():
            print ('{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(), df_scores[metric].std()))
    
    return

if __name__ == '__main__':
    main()