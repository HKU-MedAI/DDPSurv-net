import argparse
import numpy as np
from dspm_dataset import support, synthetic, kkbox, mimic
from baseline import baseline_fn 




def result(n_run, model, dataset, lr, k1, k2, epoch, eta, edit_censor, censor_rate):
    cis_list , brs_list, roc_aoc_list = [], [], []
    for j in range(n_run):
        seed = 42 + j 
        cis, brs, roc_aoc = baseline_fn(model, dataset, lr, k1+k2, k2, seed, epoch, eta, edit_censor, censor_rate)
        cis_list.append(cis)
        brs_list.append(brs)
        roc_aoc_list.append(roc_aoc)
        cis_mean = sum(cis_list) / n_run
        brs_mean = sum(brs_list) / n_run
        roc_aoc_mean = sum(roc_aoc_list) / n_run

    return cis_mean, brs_mean , roc_aoc_mean


def ablation_study(n_run, k1_range, k2_range, dataset, step_1, step_2, lr, epoch, eta):
    cis_dict, brs_dict, roc_auc_dict = {}, {}, {}
    for i in range(1, k1_range+1, step_1):
        for j in range(1, k2_range+1, step_2):
            print(i, j)
            cis, brs, roc_auc = result(n_run, 'DDPSM', dataset , lr, i+j , j, 42, epoch, eta, False, 0.9)
            cis_dict[(i,j)] = cis
            brs_dict[(i,j)] = brs
            roc_auc_dict[(i,j)] = roc_auc
    return cis_dict, brs_dict, roc_auc_dict

import matplotlib.pyplot as plt

def plot_k(dict, title, k1_list, k2_list, quantile):
    plt.figure(figsize=(8,6))
    x = k2_list
    plt.xlabel('k2')
    plt.ylabel(title)
    plt.title(title + ' for different k1 and k2')
    for k1 in k1_list:
        y = []
        for key in dict.keys():
            if key[0] == k1:
                y.append(dict[key][quantile])   
        plt.plot(x, y, label='k1='+str(k1))
    plt.legend(bbox_to_anchor=(1.05,0.5), loc=6)
    plt.savefig(title+'.png', bbox_inches='tight')
    plt.show()
    pass

import seaborn as sns
import pandas as pd

def heatmap(dict, title, k1_list, k2_list, quantile):
    result = np.zeros((len(k1_list), len(k2_list)))
    for key in dict.keys():
        k1 = key[0]
        k2 = key[1]
        if (k1 in k1_list) and (k2 in k2_list):
            k1_idx = k1_list.index(k1)
            k2_idx = k2_list.index(k2)
            result[k1_idx][k2_idx] = dict[key][quantile]
    df = pd.DataFrame(result, index=k1_list[::-1], columns=k2_list)
    print(df)
    fig = sns.heatmap(df, annot=False, cmap='hot_r', robust=True)
    fig.set_title(title)
    fig.set_xlabel('k2')
    fig.set_ylabel('k1')
    heatmap_fig = fig.get_figure()
    filepath = '/home/r10user10/Documents/Jiacheng/dspm-auton-survival'
    heatmap_fig.savefig(filepath + '/' + title + '.png')
    pass

    

        


    
        


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='ablation study hyperparameter')

    parse.add_argument('--dataset', '-d', type=str, default='support')
    parse.add_argument('--model', '-m', type=str, default='DDPSM')
    parse.add_argument('--epoch', '-e', type=int, default=100)
    parse.add_argument('--quantile', '-q', type=int, default=0)
    parse.add_argument('--lr', type=float, default=1e-4)
    parse.add_argument('--k1_range', type=int, default=2)
    parse.add_argument('--k2_range', type=int, default=1)
    parse.add_argument('--k1_step', type=int, default=1)
    parse.add_argument('--k2_step', type=int, default=1)
    parse.add_argument('--eta', type=int, default=10)
    parse.add_argument('--save', type=bool, default=True)
    parse.add_argument('--print', type=bool, default=True)
    parse.add_argument('--n_run', type=int, default=3)
    args = parse.parse_args()


    cis_dict, brs_dict, roc_auc_dict = ablation_study(args.n_run, args.k1_range, args.k2_range, args.dataset, args.k1_step, args.k2_step, args.lr, args.epoch, args.eta)
    k1_list = range(1, args.k1_range+1, args.k1_step)
    k2_list = range(1, args.k2_range+1, args.k2_step)

    if args.print:
        plot_k(cis_dict, args.dataset+' C index', k1_list, k2_list, args.quantile)
        print('print successfully')

    if args.save:
        np.save(args.dataset+'_cis_dict.npy', cis_dict)
        np.save(args.dataset+'_brs_dict.npy', brs_dict)
        np.save(args.dataset+'_roc_auc_dict.npy', roc_auc_dict)
        print('save successfully')

    sum_index = (1,1)
    for key in cis_dict.keys():
        if sum(cis_dict[key]) > sum(cis_dict[sum_index]):
            sum_index = key
    print(sum_index)

    print(cis_dict[sum_index])
    print(brs_dict[sum_index])
    print(roc_auc_dict[sum_index])
    print(args.lr, args.dataset, args.k1_step)
