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

# %%

import time
import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings("ignore")

from self_paced_ensemble import SelfPacedEnsembleClassifier
from self_paced_ensemble.canonical_ensemble import *
from self_paced_ensemble.utils import *
import argparse
from tqdm import trange

METHODS = ['SPEnsemble', 'SMOTEBoost', 'SMOTEBagging', 'RUSBoost', 'UnderBagging', 'Cascade']
RANDOM_STATE = 42

def parse():
    '''Parse system arguments.'''
    parser = argparse.ArgumentParser(
        description='Self-paced Ensemble', 
        usage='run_example.py --method <method> --n_estimators <integer> --runs <integer>'
        )
    parser.add_argument('--method', type=str, default='SPEnsemble', 
        choices=METHODS+['all'], help='Name of ensmeble method')
    parser.add_argument('--n_estimators', type=int, default=10, help='Number of base estimators')
    parser.add_argument('--runs', type=int, default=10, help='Number of independent runs')
    return parser.parse_args()

def init_model(method, base_estimator, n_estimators):
    '''return a model specified by "method".'''
    if method == 'SPEnsemble':
        model = SelfPacedEnsembleClassifier(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'SMOTEBoost':
        model = SMOTEBoostClassifier(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'SMOTEBagging':
        model = SMOTEBaggingClassifier(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'RUSBoost':
        model = RUSBoostClassifier(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'UnderBagging':
        model = UnderBaggingClassifier(base_estimator = base_estimator, n_estimators = n_estimators)
    elif method == 'Cascade':
        model = BalanceCascadeClassifier(base_estimator = base_estimator, n_estimators = n_estimators)
    else:
        raise ValueError(f'Do not support method {method}. Only accept \
            \'SPEnsemble\', \'SMOTEBoost\', \'SMOTEBagging\', \'RUSBoost\', \
            \'UnderBagging\', \'Cascade\'.')
    return model

def main():
    # Parse arguments
    args = parse()
    method_used = args.method
    n_estimators = args.n_estimators
    runs = args.runs

    # Load train/test data
    X_train, X_test, y_train, y_test = load_covtype_dataset(
        subset=0.1, random_state=RANDOM_STATE)

    # Train & Record
    method_list = METHODS if method_used=='all' else [method_used]
    for method in method_list:
        print('\nRunning method:\t\t{} - {} estimators in {} independent run(s) ...'.format(
            method, n_estimators, runs))
        # print('Running ...')
        scores = []; times = []
        try:
            with trange(runs, ncols=80) as t:
                t.set_description(f'{method} running')
                for _ in t:
                    model = init_model(
                        method=method,
                        n_estimators=n_estimators,
                        base_estimator=sklearn.tree.DecisionTreeClassifier(),
                    )
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    times.append(time.time()-start_time)
                    y_pred = model.predict_proba(X_test)[:, 1]
                    scores.append([
                        auc_prc(y_test, y_pred),
                        f1_optim(y_test, y_pred),
                        gm_optim(y_test, y_pred),
                        mcc_optim(y_test, y_pred)
                    ])
        except KeyboardInterrupt:
            t.close()
        
        # Print results to console
        print('ave_run_time:\t\t{:.3f}s'.format(np.mean(times)))
        print('------------------------------')
        print('Metrics:')
        df_scores = pd.DataFrame(scores, columns=['AUCPRC', 'F1', 'G-mean', 'MCC'])
        for metric in df_scores.columns.tolist():
            print ('{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(), df_scores[metric].std()))
    
    return

if __name__ == '__main__':
    main()
# %%
