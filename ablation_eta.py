import argparse
import numpy as np
import matplotlib.pyplot as plt
from dspm_dataset import support, synthetic, kkbox, mimic3, mimic4, metabric
from baseline import baseline_fn 
import os




def result(n_run, model, dataset, lr, k1, k2, epoch, eta, edit_censor, censor_rate, dist, plot_dist = False):
    cis_list , brs_list, roc_aoc_list = [], [], []
    for j in range(n_run):
        seed = 42 + j 
        cis, brs, roc_aoc = baseline_fn(model, dataset, lr, k1+k2, k2, seed, epoch, eta, edit_censor, censor_rate, dist, plot_dist)
        cis_list.append(cis)
        brs_list.append(brs)
        roc_aoc_list.append(roc_aoc)
        cis_mean = np.mean(np.array(cis_list), axis=0)
        # print(cis_mean.shape)
        brs_mean = np.mean(np.array(brs_list), axis=0)
        roc_aoc_mean = np.mean(np.array(roc_aoc_list), axis=0)

    return cis_mean, brs_mean , roc_aoc_mean

def ablation_eta(n_run, dataset, k1, k2, lr, epoch, eta_list):
    cis_dict, brs_dict, roc_auc_dict = {}, {}, {}
    for eta in eta_list:
        cis, brs, roc_auc = result(n_run, 'DDPSM', dataset , lr, k1+k2 , k2, epoch, eta, False, 0.9, 'LogNormal', False)
        cis_dict[eta] = cis
        brs_dict[eta] = brs
        roc_auc_dict[eta] = roc_auc
    return cis_dict, brs_dict, roc_auc_dict


def plot_eta(dict, eta_list, dirpath):
    horizon = [0.25, 0.5, 0.75, 0.9]
    plt.figure(figsize=(8,6))
    x = eta_list
    plt.xlabel('eta')
    plt.ylabel('C index')
    plt.title('C index for different eta')
    plt.xscale('log')
    for quantile in range(4):
        y = []
        for eta in eta_list:
            y.append(dict[eta][quantile])   
        plt.plot(x, y, label='horizon='+str(horizon[quantile]))
    plt.legend(bbox_to_anchor=(1.05,0.5), loc=6)
    plt.savefig(dirpath + '/' + 'eta.png', bbox_inches='tight')
    plt.show()
    pass
    
        


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='ablation study hyperparameter')

    parse.add_argument('--dataset', '-d', type=str, default='support')
    parse.add_argument('--epoch', '-e', type=int, default=100)
    parse.add_argument('--quantile', '-q', type=int, default=0)
    parse.add_argument('--lr', type=float, default=1e-4)
    parse.add_argument('--k1', type=int, default=2)
    parse.add_argument('--k2', type=int, default=1)
    parse.add_argument('--save', '-s', type=bool, default=True)
    parse.add_argument('--print', '-p', type=bool, default=True)
    parse.add_argument('--n_run', type=int, default=1)
    args = parse.parse_args()


    cis_dict, brs_dict, roc_auc_dict = ablation_eta(args.n_run, args.dataset, args.k1, args.k2, args.lr, args.epoch, [0.01, 0.1, 10 ,100, 1000])
    dirpath = '/home/r10user10/Documents/Jiacheng/dspm-auton-survival/ablation_study'+'/'+args.dataset
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    if args.print:
        plot_eta(cis_dict, [0.01, 0.1, 10 ,100, 1000], dirpath)
        print('print successfully')

    if args.save:
        np.save(dirpath+'/'+f'{args.k1}_{args.k2}_'+'eta_cis_dict.npy', cis_dict)
        np.save(dirpath+'/'+f'{args.k1}_{args.k2}_'+'eta_brs_dict.npy',brs_dict)
        np.save(dirpath+'/'+f'{args.k1}_{args.k2}_'+'eta_roc_auc_dict.npy', roc_auc_dict)
        # np.save(dirpath+'/'+args.dataset+'_cis_dict.npy', cis_dict)
        # np.save(dirpath+'/'+args.dataset+'_brs_dict.npy', brs_dict)
        # np.save(dirpath+'/'+args.dataset+'_roc_auc_dict.npy', roc_auc_dict)
        print('save npy successfully')

    sum_index = 0.1
    for key in cis_dict.keys():
        if sum(cis_dict[key]) > sum(cis_dict[sum_index]):
            sum_index = key
    print(sum_index)

    print(cis_dict[sum_index])
    print(brs_dict[sum_index])
    print(roc_auc_dict[sum_index])
    print(args.dataset, args.lr, args.k1, args.k2)