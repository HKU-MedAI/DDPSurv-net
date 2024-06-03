import argparse
import numpy as np
import matplotlib.pyplot as plt
from dspm_dataset import support, synthetic, kkbox, mimic
from baseline import baseline_fn 



def ablation_eta(dataset, k1, k2, lr, epoch, eta_list):
    cis_dict, brs_dict, roc_auc_dict = {}, {}, {}
    for eta in eta_list:
        cis, brs, roc_auc = baseline_fn('DDPSM', dataset , lr, k1+k2 , k2, 42, epoch, eta, False, 0.9)
        cis_dict[eta] = cis
        brs_dict[eta] = brs
        roc_auc_dict[eta] = roc_auc
    return cis_dict, brs_dict, roc_auc_dict


# def plot_k(dict, title, eta_list, quantile):
#     plt.figure(figsize=(8,6))
#     x = eta_list
#     plt.xlabel('eta')
#     plt.ylabel(title)
#     plt.title(title + ' for different k1 and k2')
#     for eta in eta_list:
#         y = []
#         for key in dict.keys():
#             if key[0] == k1:
#                 y.append(dict[key][quantile])   
#         plt.plot(x, y, label='k1='+str(k1))
#     plt.legend(bbox_to_anchor=(1.05,0.5), loc=6)
#     plt.savefig(title+'.png', bbox_inches='tight')
#     plt.show()
#     pass
    
        


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='ablation study hyperparameter')

    parse.add_argument('--dataset', '-d', type=str, default='support')
    parse.add_argument('--model', '-m', type=str, default='DDPSM')
    parse.add_argument('--epoch', '-e', type=int, default=100)
    parse.add_argument('--quantile', '-q', type=int, default=0)
    parse.add_argument('--lr', type=float, default=1e-4)
    parse.add_argument('--k1', type=int, default=2)
    parse.add_argument('--k2', type=int, default=1)
    parse.add_argument('--save', type=bool, default=True)
    parse.add_argument('--print', type=bool, default=True)
    args = parse.parse_args()


    cis_dict, brs_dict, roc_auc_dict = ablation_eta(args.dataset, args.k1, args.k2, args.lr, args.epoch, [0.01, 0.1, 1, 10 ,100, 1000])

    # if args.print:
    #     plot_k(cis_dict, args.dataset+' C index', k1_list, k2_list, args.quantile)

    if args.save:
        np.save(args.dataset+'_eta_cis_dict.npy', cis_dict)
        np.save(args.dataset+'_eta_brs_dict.npy', brs_dict)
        np.save(args.dataset+'_eta_roc_auc_dict.npy', roc_auc_dict)

    sum_index = 0.1
    for key in cis_dict.keys():
        if sum(cis_dict[key]) > sum(cis_dict[sum_index]):
            sum_index = key
    print(sum_index)

    print(cis_dict[sum_index])
    print(brs_dict[sum_index])
    print(roc_auc_dict[sum_index])
    print(args.dataset, args.lr, args.k1, args.k2)