import argparse
import numpy as np
import matplotlib.pyplot as plt
from dspm_dataset import support, synthetic, kkbox, mimic3, mimic4, metabric
from baseline import baseline_fn 
import os
import pandas as pd




def result(n_run, model, dataset, lr, k1, k2, epoch, eta, edit_censor, censor_rate, dist):
    cis_list , brs_list, roc_aoc_list = [], [], []
    for j in range(n_run):
        seed = 42 + j 
        cis, brs, roc_aoc = baseline_fn(model, dataset, lr, k1+k2, k2, seed, epoch, eta, edit_censor, censor_rate, dist)
        cis_list.append(cis)
        brs_list.append(brs)
        roc_aoc_list.append(roc_aoc)
    cis_mean = np.mean(np.array(cis_list))
    # print(cis_mean.shape)
    brs_mean = np.mean(np.array(brs_list))
    roc_aoc_mean = np.mean(np.array(roc_aoc_list))

    return cis_mean, brs_mean , roc_aoc_mean

def ablation_censor(model, n_run, dataset, k1, k2, lr, epoch, censor_list, dist):
    cis_dict, brs_dict, roc_auc_dict = {}, {}, {}
    for cr in censor_list:
        cis, brs, roc_auc = result(n_run, model, dataset , lr, k1 , k2, epoch, 10, True, cr, dist)
        cis_dict[cr] = cis
        brs_dict[cr] = brs
        roc_auc_dict[cr] = roc_auc
    cis_def, brs_def, roc_auc_def = result(n_run, model, dataset , lr, k1 , k2, epoch, 10, False, 0.9, dist)
    cis_dict['default'] = cis_def
    brs_dict['default'] = brs_def
    roc_auc_dict['default'] = roc_auc_def
    return cis_dict, brs_dict, roc_auc_dict
   
        


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='ablation study hyperparameter')

    parse.add_argument('--model', '-m', type=str, default='DDPSM')
    parse.add_argument('--dataset', '-d', type=str, default='support')
    parse.add_argument('--epoch', '-e', type=int, default=200)
    parse.add_argument('--quantile', '-q', type=int, default=0)
    parse.add_argument('--lr', type=float, default=1e-4)
    parse.add_argument('--k1', type=int, default=2)
    parse.add_argument('--k2', type=int, default=1)
    parse.add_argument('--save', '-s', type=bool, default=True)
    parse.add_argument('--print', '-p', type=bool, default=True)
    parse.add_argument('--n_run', type=int, default=1)
    parse.add_argument('--dist', type=str, default='LogNormal')
    args = parse.parse_args()

    censor_rate_list = [(0.1* i + 0.4) for i in range(6)]

    dirpath = f'/home/r10user10/Documents/Jiacheng/dspm-auton-survival/ablation_study/{args.dataset}/'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


    # save cis_dict to one single csv file where key as column name
    file_name = dirpath + f'{args.k1}_{args.k2}_censor.csv'

    model_list = ['DDPSM', 'DSM', 'DCM']
    df_cis = pd.DataFrame()
    for model_name in model_list:
        cis_dict, brs_dict, roc_auc_dict = ablation_censor(model_name, args.n_run, args.dataset, args.k1, args.k2, args.lr, args.epoch, censor_rate_list, args.dist)
        # let cis_dict.keys() be the column name, cis_dict.values() be the value, model name as the index
        df_cis = df_cis.append(pd.DataFrame(cis_dict, index=[model_name]))
    
    df_cis.to_csv(file_name)

    

