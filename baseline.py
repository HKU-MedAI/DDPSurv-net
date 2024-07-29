import argparse
from dspm_dataset import support, synthetic, kkbox, mimic4, mimic3, metabric, edit_censor_rate
import numpy as np
import pandas as pd
import numpy as np
from auton_survival import DeepCoxPH
from auton_survival.models.dsm import DeepSurvivalMachines
from auton_survival.models.dcm import DeepCoxMixtures
from auton_survival.models.dpsm import DeepDP
import os
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from distribution import weibull, lognormal, logcauchy
import matplotlib.pyplot as plt
# from DeepHit.summarize_results import run_DeepHit 


DSM_model = ['DeepCox', 'DSM', 'DCM', 'DDPSM']

def baseline_fn(baseline, dataset, lr, n_components, n_cauchy, seed, epoch, eta, edit_censor, censor_rate, dist):
    if baseline == 'DeepCox':
        model = DeepCoxPH(layers=[100,100], random_seed = seed)
    if baseline == 'DSM':
        model = DeepSurvivalMachines(k = n_components,
                                distribution = dist,
                                layers = [100,100], random_seed = seed)
    if baseline == 'DCM':
        model = DeepCoxMixtures(k = n_components, layers = [100,100], random_seed = seed)
    if baseline == 'DDPSM':
        model = DeepDP(k= n_components, k2 = n_cauchy, eta=eta,
               distribution= dist,
               layers=[100,100], random_seed = seed)

    if dataset == 'support':
        x_train, t_train , e_train, x_test, t_test , e_test = support()
        # print(x_train.shape[-1])
    if dataset == 'synthetic':
        x_train, t_train , e_train, x_test, t_test , e_test = synthetic()
        # print(x_train.shape[-1])
    if dataset == 'kkbox':
        x_train, t_train , e_train, x_test, t_test , e_test = kkbox()
        # print(x_train.shape[-1])
    if dataset == 'mimic4':
        x_train, t_train , e_train, x_test, t_test , e_test = mimic4()  
        # print(x_train.shape[-1])
    if dataset == 'mimic3':
        x_train, t_train , e_train, x_test, t_test , e_test = mimic3()
    if dataset == 'metabric':
        x_train, t_train , e_train, x_test, t_test , e_test = metabric()


    if edit_censor:
        t_train, e_train = edit_censor_rate(censor_rate, e_train, t_train, 'fix')
        t_test, e_test = edit_censor_rate(censor_rate, e_test, t_test, 'fix')
        print('edit censor rate success')
    


    model.fit(x_train, t_train, e_train, iters = epoch, learning_rate = lr)
    shape, scale = model.show_distribution_params(x_train, risk='1')
    shape = shape.cpu().detach().numpy().mean(axis=0)
    scale = scale.cpu().detach().numpy().mean(axis=0)
    # print(f"Shape: {shape}, Scale: {scale}")
    # print(shape.shape, scale.shape)
    if args.plot_dist:
        x_plot = np.linspace(0.01, 10, 5000)
        for i in range(model.k - model.k2):
            if dist == 'Weibull':
                y_plot = weibull(x_plot, np.abs(shape[i]), np.exp(scale[i]))
                # import ipdb
                # ipdb.set_trace()
            elif dist == 'LogNormal':
                y_plot = lognormal(x_plot, np.abs(shape[i]), np.exp(scale[i]))
            plt.plot(x_plot, y_plot, label=f'Component {i}')
        for j in range(model.k2):
            y_plot = logcauchy(x_plot, np.abs(shape[model.k - model.k2 + j]), np.exp(scale[model.k - model.k2 + j]))
            plt.plot(x_plot, y_plot, label=f'Component {model.k - model.k2 + j}')
        plt.legend()
        plt.show()
        plt.savefig(f'./results/{args.dataset}/{args.k1}_{args.k2}_distribution.png')

    



    horizons = [0.25, 0.5, 0.75, 0.9]
    x = np.concatenate((x_train, x_test), axis=0)
    t = np.concatenate((t_train, t_test), axis=0)
    e = np.concatenate((e_train, e_test), axis=0)
    times = np.quantile(t[e==1], horizons).tolist()
    out_risk = 1 - model.predict_survival(x_test, times)
    out_survival = model.predict_survival(x_test, times)
    # print(out_survival.shape)
    # t_graph = np.linspace(t[e==1].min(), t[e==0].max(), 1000).tolist()
    # tr_graph = model.predict_survival(x_train, t_graph)
    # te_graph = model.predict_survival(x_test, t_graph)
    # graph_x = t_graph
    # graph_y = tr_graph[0,:]
    # plt.plot(graph_x, graph_y, label='train_0')
    # plt.show()
    # plt.savefig('survival_distribution.png')

    # import ipdb
    # ipdb.set_trace()


    cis = []
    brs = []

    et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                    dtype = [('e', bool), ('t', float)])
    et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                    dtype = [('e', bool), ('t', float)])
    # et_val = np.array([(e_val[i], t_val[i]) for i in range(len(e_val))],
    #                  dtype = [('e', bool), ('t', float)])
    # print(et_train[0:10])
    for i, _ in enumerate(times):
        cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])
    brs.append(brier_score(et_train, et_test, out_survival, times)[1])
    roc_auc = []
    for i, _ in enumerate(times):
        roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0])
    # for horizon in enumerate(horizons):
    #     print(f"For {horizon[1]} quantile")
    #     print("TD Concordance Index:", cis[horizon[0]])
    #     print("Brier Score:", brs[0][horizon[0]])
    #     print("ROC AUC ", roc_auc[horizon[0]][0], "\n")
    return cis, brs, roc_auc
    

def result(n_run, model, dataset, lr, k1, k2, epoch, eta, edit_censor, censor_rate, dist):
    cis_list , brs_list, roc_aoc_list = [], [], []
    for j in range(n_run):
        seed = 42 + j 
        cis, brs, roc_aoc = baseline_fn(model, dataset, lr, k1+k2, k2, seed, epoch, eta, edit_censor, censor_rate, dist)
        cis_list.append(cis)
        brs_list.append(brs)
        roc_aoc_list.append(roc_aoc)

    report_str = f"""
    Results:
        mean c-index: {np.asarray(cis_list).mean(axis=0)}
        std c-index: {np.asarray(cis_list).std(axis=0)}
        mean bs: {np.asarray(brs_list).mean(axis=0)}
        std bs: {np.asarray(brs_list).std(axis=0)}
        mean roc_aoc: {np.asarray(roc_aoc_list).mean(axis=0)}
        std roc_aoc: {np.asarray(roc_aoc_list).std(axis=0)}
    """
    print(report_str)
    return cis_list, brs_list , roc_aoc_list

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='Baseline hyperparameter')

    parse.add_argument('--dataset', '-d', type=str, default='support')
    parse.add_argument('--model', '-m', type=str, default='DDPSM')
    parse.add_argument('--n_run', '-n', type=int, default=5)
    parse.add_argument('--epoch', '-e', type=int, default=100)
    parse.add_argument('--lr', type=float, default=1e-4)
    parse.add_argument('--censor_rate', type=float, default=0.9)
    parse.add_argument('--k1', type=int, default=2)
    parse.add_argument('--k2', type=int, default=1)
    parse.add_argument('--eta', type=float, default=10)
    parse.add_argument('--edit_censor', type=bool, default=False)
    parse.add_argument('--save_csv', type=bool, default=True)
    parse.add_argument('--dist', type=str, default='Weibull')
    parse.add_argument('--plot_dist', type=bool, default=True)
    args = parse.parse_args()


    # cis_list, brs_list , roc_aoc_list = result(10, 'DDPSM', 'mimic', 1e-4, 15, 5)
    if args.model in DSM_model:
        cis_list, brs_list , roc_aoc_list = result(args.n_run, args.model, args.dataset , args.lr, args.k1, args.k2, args.epoch, args.eta, args.edit_censor, args.censor_rate, args.dist)
    # elif args.model == 'DeepHit':
    #     cis_list, brs_list, roc_auc_list = run_DeepHit(1234, args.dataset)
    # elif args.model == 'nfm':
    #     pass
    # elif args.model == 'Sumo-Net':  
    #     pass 
    else:
        raise ValueError("Model not found")
    
    cis_mean = np.round(np.asarray(cis_list).mean(axis=0),4).tolist()
    cis_std = np.round(np.asarray(cis_list).std(axis=0),4).tolist()
    brs_mean = np.round(np.asarray(brs_list).mean(axis=0),4).tolist()[0]
    brs_std = np.round(np.asarray(brs_list).std(axis=0),4).tolist()[0]
    print(cis_mean, cis_std, brs_mean, brs_std)

    if args.save_csv:
        df = pd.DataFrame({'mean_c-index':cis_mean, 'std_c-index': cis_std,
                          'mean_brier_score': brs_mean, 'std_brier_score': brs_std}, index=[0.25, 0.5, 0.75, 0.9])
        dir_path = f'./results/{args.dataset}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_path = dir_path + f'{args.model}_{args.dataset}_{args.lr}_{args.k1}_{args.k2}.csv'
        df.to_csv(save_path, index=False)
        print(f"Save to {args.model}_{args.dataset}.csv")



