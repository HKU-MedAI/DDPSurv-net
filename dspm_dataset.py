import random
import numpy as np
from pycox.datasets import from_deepsurv, from_kkbox

def support():
    import sys
    sys.path.append('../')
    from auton_survival import datasets
    outcomes, features = datasets.load_support()
    from auton_survival.preprocessing import Preprocessor
    cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
    num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
            'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
                'glucose', 'bun', 'urine', 'adlp', 'adls']

    features = Preprocessor().fit_transform(features, cat_feats=cat_feats, num_feats=num_feats)
    x, t, e = features.values, outcomes.time.values, outcomes.event.values

    n = len(x)

    tr_size = int(n*0.80)
    te_size = int(n*0.20)

    x_train, x_test = x[:tr_size], x[-te_size:]
    t_train, t_test = t[:tr_size], t[-te_size:]
    e_train, e_test = e[:tr_size], e[-te_size:]
    return x_train, t_train , e_train, x_test, t_test , e_test


def synthetic():
    import pandas as pd
    import torch
    from tqdm import tqdm 
    import sys
    sys.path.append('../')

    from auton_survival.datasets import load_dataset

    # Load the synthetic dataset
    outcomes, features, interventions = load_dataset(dataset='SYNTHETIC')

    # Hyper-parameters
    random_seed = 0
    test_size = 0.25

    # Split the synthetic data into training and testing data
    import numpy as np

    np.random.seed(random_seed)
    n = features.shape[0] 

    test_idx = np.zeros(n).astype('bool')
    test_idx[np.random.randint(n, size=int(n*test_size))] = True 

    features_tr = features.iloc[~test_idx] 
    outcomes_tr = outcomes.iloc[~test_idx]
    interventions_tr = interventions[~test_idx]
    # print(f'Number of training data points: {len(features_tr)}')

    features_te = features.iloc[test_idx] 
    outcomes_te = outcomes.iloc[test_idx]
    interventions_te = interventions[test_idx]
    # print(f'Number of test data points: {len(features_te)}')

    interventions_tr.name, interventions_te.name = 'treat', 'treat'
    features_tr_dcph = pd.concat([features_tr, interventions_tr.astype('float64')], axis=1)
    features_te_dcph = pd.concat([features_te, interventions_te.astype('float64')], axis=1)
    outcomes_tr_dcph = pd.DataFrame(outcomes_tr, columns=['event', 'time']).astype('float64')


    x_train = features_tr_dcph.values
    e_train = outcomes_tr['event'].values.astype(float)
    t_train = outcomes_tr['time'].values

    x_test = features_te_dcph.values
    e_test = outcomes_te['event'].values.astype(float)
    t_test = outcomes_te['time'].values

    print(x_train.dtype, t_train.dtype, e_train.dtype)
    return x_train, t_train , e_train, x_test, t_test , e_test

def kkbox():

    kkbox_data = from_kkbox._DatasetKKBoxChurn()
    #kkbox_data.download_kkbox()

    df = kkbox_data.read_df()

    import numpy as np
    import pandas as pd

    e = np.array(df.event)
    t = np.array(df.duration)
    x = df.drop(columns=['event','duration','msno'])

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    x['gender'] = le.fit_transform(x['gender'])
    x['registered_via'] = le.fit_transform(x['registered_via'])
    x['city'] = le.fit_transform(x['city'])
    x = np.array(x).astype(float)

    import os, sys
    import numpy as np 

    # path = '/home/r10user10/Documents/Jiacheng/dspm-auton-survival'
    # os.chdir(path)
    # print(os.getcwd())

    from auton_survival import datasets

    n = len(x)

    tr_size = int(n * 0.80)
    te_size = int(n * 0.20)


    x_train, x_test = x[:tr_size], x[-te_size:]
    t_train, t_test = t[:tr_size], t[-te_size:]
    e_train, e_test = e[:tr_size], e[-te_size:]
    return x_train, t_train , e_train, x_test, t_test , e_test


def mimic4():

    x_train = np.load('datasets/mimic4/mimic4_x_train.npy')
    t_train = np.load('datasets/mimic4/mimic4_t_train.npy')
    e_train = 1 - np.load('datasets/mimic4/mimic4_e_train.npy')
    index = np.where(t_train <= 0)[0]
    t_train = np.delete(t_train, index)
    e_train = np.delete(e_train, index)
    x_train = np.delete(x_train, index, axis=0)
    x_test = np.load('datasets/mimic4/mimic4_x_test.npy')
    t_test = np.load('datasets/mimic4/mimic4_t_test.npy')
    e_test = 1 - np.load('datasets/mimic4/mimic4_e_test.npy')
    index = np.where(t_test <= 0)[0]
    t_test = np.delete(t_test, index)
    e_test = np.delete(e_test, index)
    x_test = np.delete(x_test, index, axis=0)
    x_train = np.mean(x_train, axis=1)
    x_test = np.mean(x_test, axis=1)
    print(x_train.shape)
    return x_train, t_train , e_train, x_test, t_test , e_test

def mimic3():
    x_train = np.load('datasets/mimic3/mimic3_x_train.npy')
    t_train = np.load('datasets/mimic3/mimic3_t_train.npy')
    e_train = 1 - np.load('datasets/mimic3/mimic3_e_train.npy')
    index = np.where(t_train <= 0)[0]
    t_train = np.delete(t_train, index)
    e_train = np.delete(e_train, index)
    x_train = np.delete(x_train, index, axis=0)
    x_test = np.load('datasets/mimic3/mimic3_x_test.npy')
    t_test = np.load('datasets/mimic3/mimic3_t_test.npy')
    e_test = 1 - np.load('datasets/mimic3/mimic3_e_test.npy')
    index = np.where(t_test <= 0)[0]
    t_test = np.delete(t_test, index)
    e_test = np.delete(e_test, index)
    x_test = np.delete(x_test, index, axis=0)
    x_train = np.mean(x_train, axis=1)
    x_test = np.mean(x_test, axis=1)
    print(x_train.shape)
    return x_train, t_train , e_train, x_test, t_test , e_test
    

def metabric():
    data = from_deepsurv._Metabric()
    data_full = data.read_df()

    x = data_full.drop(columns=['duration', 'event']).values.astype('float64')
    t = data_full['duration'].values.astype('float64')
    e = data_full['event'].values.astype('float64')
    
    print(x.dtype, t.dtype, e.dtype)

    n = len(x)

    tr_size = int(n*0.80)
    te_size = int(n*0.20)

    x_train, x_test = x[:tr_size], x[-te_size:]
    t_train, t_test = t[:tr_size], t[-te_size:]
    e_train, e_test = e[:tr_size], e[-te_size:]

    return x_train, t_train , e_train, x_test, t_test , e_test


def edit_censor_rate(censor_rate, e, t, method):
    n_total = e.shape[0]
    default_censor_idx = np.where(e==0)[0]
    default_non_censor_idx = np.where(e==1)[0]
    n_default = default_censor_idx.shape[0]
    n_set = int(censor_rate * n_total)
    t_min = np.min(t)
    
    if method == 'random':
        censor_index = np.random.choice(range(n_total), n_set, replace=False)
        e = np.ones_like(e)
        e[censor_index] = 0
    if method == 'fix':
        n_lack = n_set - n_default
        print(n_set, n_default, n_lack)
        assert n_lack > 0
        new_censor_index = np.random.choice(default_non_censor_idx, n_lack, replace=False)
        e[new_censor_index] = 0
        for idx in new_censor_index:
            t_org = t[idx]
            # print(t_org, t_min)
            t[idx] = np.random.uniform(t_min, t_org)
        
    return t, e

