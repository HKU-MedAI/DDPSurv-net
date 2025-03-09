import argparse
import numpy as np
from dspm_dataset import support, synthetic, kkbox, mimic3, mimic4, metabric
from baseline import baseline_fn 
import os
import wandb




def result(n_run, model, dataset, lr, k1, k2, epoch, eta, edit_censor, censor_rate):
    cis_list , brs_list, roc_aoc_list = [], [], []
    for j in range(n_run):
        seed = 42 + j 
        cis, brs, roc_aoc = baseline_fn(model, dataset, lr, k1+k2, k2, seed, epoch, eta, edit_censor, censor_rate, 'LogNormal', False)
        cis_list.append(cis)
        brs_list.append(brs)
        roc_aoc_list.append(roc_aoc)
        cis_mean = np.mean(np.array(cis_list), axis=0)
        brs_mean = np.mean(np.array(brs_list), axis=0)
        roc_aoc_mean = np.mean(np.array(roc_aoc_list), axis=0)
        cis_std = np.std(np.array(cis_list), axis=0)
        brs_std = np.std(np.array(brs_list), axis=0)
        roc_aoc_std = np.std(np.array(roc_aoc_list), axis=0)

    return cis_mean, brs_mean , roc_aoc_mean, cis_std, brs_std, roc_aoc_std


def ablation_study(n_run, k1_range, k2_range, dataset, step_1, step_2, lr, epoch, eta):
    cis_mean_dict, brs_mean_dict, roc_auc_mean_dict = {}, {}, {}
    cis_std_dict, brs_std_dict, roc_auc_std_dict = {}, {}, {}
    for i in range(1, k1_range+1, step_1):
        for j in range(0, k2_range, step_2):
            wandb.init(entity="", project=f'dspm_{dataset}', name=f'{lr}_{i}_{j}')
            wandb.config = {'k1': i, 'k2': j}
            print(i, j)
            cis_mean, brs_mean, roc_auc_mean, cis_std, brs_std, roc_aoc_std = result(n_run, 'DDPSM', dataset , lr, i , j, epoch, eta, False, 0.9)
            cis_mean_dict[(i,j)] = cis_mean
            brs_mean_dict[(i,j)] = brs_mean
            roc_auc_mean_dict[(i,j)] = roc_auc_mean
            cis_std_dict[(i,j)] = cis_std
            brs_std_dict[(i,j)] = brs_std
            roc_auc_std_dict[(i,j)] = roc_aoc_std
            wandb.log({'c-index_25': cis_mean[0], 'brier_score_25': brs_mean[0][0], 
                       'c-index_50': cis_mean[1], 'brier_score_50': brs_mean[0][1],
                        'c-index_75': cis_mean[2], 'brier_score_75': brs_mean[0][2],
                        'c-index_90': cis_mean[3], 'brier_score_90': brs_mean[0][3]
                       })
            wandb.finish()
    return cis_mean_dict, brs_mean_dict, roc_auc_mean_dict, cis_std_dict, brs_std_dict, roc_auc_std_dict

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

def heatmap(dict, title, k1_list, k2_list, quantile, path):
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
    fig.set_title(title + ' heatmap C index')
    fig.set_xlabel('k2')
    fig.set_ylabel('k1')
    heatmap_fig = fig.get_figure()
    heatmap_fig.savefig(path + '/' + title + ' heatmap C index.png')
    pass

    

        


    
        


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='ablation study hyperparameter')

    parse.add_argument('--dataset', '-d', type=str, default='synthetic')
    parse.add_argument('--model', '-m', type=str, default='DDPSM')
    parse.add_argument('--epoch', '-e', type=int, default=200)
    parse.add_argument('--quantile', '-q', type=int, default=0)
    parse.add_argument('--lr', type=float, default=1e-4)
    parse.add_argument('--k1_range', type=int, default=2)
    parse.add_argument('--k2_range', type=int, default=1)
    parse.add_argument('--k1_step', type=int, default=1)
    parse.add_argument('--k2_step', type=int, default=1)
    parse.add_argument('--eta', type=int, default=10)
    parse.add_argument('--save', '-s', type=bool, default=True)
    parse.add_argument('--print', '-p', type=bool, default=True)
    parse.add_argument('--save_csv', type=bool, default=True)
    parse.add_argument('--n_run', type=int, default=1)
    args = parse.parse_args()


    cis_mean_dict, brs_mean_dict, roc_auc_mean_dict, cis_std_dict, brs_std_dict, roc_auc_std_dict  = ablation_study(args.n_run, args.k1_range, args.k2_range, args.dataset, args.k1_step, args.k2_step, args.lr, args.epoch, args.eta)
    # print_k1_list = range(1, args.k1_range+1, args.k1_step)
    # print_k2_list = range(1, args.k2_range+1, 2)
    dirpath = '/home/r10user10/Documents/Jiacheng/dspm-auton-survival/ablation_study'+'/'+args.dataset
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # if args.print:
    #     heatmap(cis_mean_dict, f'{args.dataset}_{args.lr}', print_k1_list, print_k2_list, args.quantile, dirpath)
    #     print('print successfully')

    if args.save:
        np.save(dirpath+'/'+args.dataset+'_cis_mean_dict.npy', cis_mean_dict)
        np.save(dirpath+'/'+args.dataset+'_brs_mean_dict.npy',brs_mean_dict)
        np.save(dirpath+'/'+args.dataset+'_roc_auc_mean_dict.npy', roc_auc_mean_dict)
        np.save(dirpath+'/'+args.dataset+'_cis_std_dict.npy', cis_std_dict)
        np.save(dirpath+'/'+args.dataset+'_brs_std_dict.npy', brs_std_dict)
        np.save(dirpath+'/'+args.dataset+'_roc_auc_std_dict.npy', roc_auc_std_dict)
        print('save npy successfully')

    # sum_index = (1,1)
    # for key in cis_mean_dict.keys():
    #     if sum(cis_mean_dict[key]) > sum(cis_mean_dict[sum_index]):
    #         sum_index = key
    # print(sum_index)

    sum_index = (1,1)
    for key in cis_mean_dict.keys():
        if cis_mean_dict[key][-1] > cis_mean_dict[sum_index][-1]:
            sum_index = key
    print(sum_index)
    
    if args.save_csv:
        df = pd.DataFrame({'mean_c-index':cis_mean_dict[sum_index], 'std_c-index': cis_std_dict[sum_index],
                          'mean_brier_score': brs_mean_dict[sum_index][0], 'std_brier_score': brs_std_dict[sum_index][0]}, index=[0.25, 0.5, 0.75, 0.9])
        df.to_csv(dirpath + f'/{args.dataset}_{args.lr}_{sum_index[0]}_{sum_index[1]}.csv')

    print(cis_mean_dict[sum_index])
    print(brs_mean_dict[sum_index])
    print(roc_auc_mean_dict[sum_index])
    print(args.lr, args.dataset, args.k1_step)
