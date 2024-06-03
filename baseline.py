import argparse
from dspm_dataset import support, synthetic, kkbox, mimic, edit_censor_rate
import numpy as np



def baseline_fn(baseline, dataset, lr, n_components, n_cauchy, seed, epoch, eta, edit_censor, censor_rate):
    if baseline == 'DeepCox':
        from auton_survival import DeepCoxPH
        model = DeepCoxPH(layers=[100,100], random_seed = seed)
    if baseline == 'DSM':
        from auton_survival.models.dsm import DeepSurvivalMachines
        model = DeepSurvivalMachines(k = n_components,
                                distribution = 'LogNormal',
                                layers = [100,100], random_seed = seed)
    if baseline == 'DCM':
        from auton_survival.models.dcm import DeepCoxMixtures
        model = DeepCoxMixtures(k = n_components, layers = [100,100], random_seed = seed)
    if baseline == 'DDPSM':
        from auton_survival.models.dpsm import DeepDP
        model = DeepDP(k= n_components, k2 = n_cauchy, eta=eta,
               distribution='LogNormal',
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
    if dataset == 'mimic':
        x_train, t_train , e_train, x_test, t_test , e_test = mimic()  
        # print(x_train.shape[-1])


    # if edit_censor:
    #     e_train = edit_censor_rate(censor_rate, e_train, t_train, 'fix')
    #     e_test = edit_censor_rate(censor_rate, e_test, t_test, 'fix')
    


    model.fit(x_train, t_train, e_train, iters = epoch, learning_rate = lr)
    horizons = [0.25, 0.5, 0.75, 0.9]
    x = np.concatenate((x_train, x_test), axis=0)
    t = np.concatenate((t_train, t_test), axis=0)
    e = np.concatenate((e_train, e_test), axis=0)
    times = np.quantile(t[e==1], horizons).tolist()
    out_risk = 1 - model.predict_survival(x_test, times)
    out_survival = model.predict_survival(x_test, times)
    # print(out_survival.shape)

    from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc

    cis = []
    brs = []

    et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                    dtype = [('e', bool), ('t', float)])
    #print(et_train)
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
    

# %%
import numpy as np

# # %%
# x_train, t_train , e_train, x_test, t_test , e_test = mimic()
# e = np.concatenate((e_train, e_test), axis=0)
# n_event = np.where(e==1)[0].shape[0]
# n_censor = e.shape[0] - n_event
# print(n_event) 
# print(n_censor)
# print(n_event / e.shape[0])
# print(n_censor / e.shape[0])

# %%
# cis, brs, roc_auc = baseline('DDPSM', 'support', 1e-4, 10, 5, 42)

# %%
def result(n_run, model, dataset, lr, k1, k2, epoch, eta, edit_censor, censor_rate):
    cis_list , brs_list, roc_aoc_list = [], [], []
    for j in range(n_run):
        seed = 42 + j 
        cis, brs, roc_aoc = baseline_fn(model, dataset, lr, k1+k2, k2, seed, epoch, eta, edit_censor, censor_rate)
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
    parse.add_argument('--edit_censor', type=str, default='False')
    args = parse.parse_args()

    # %%
    # cis_list, brs_list , roc_aoc_list = result(10, 'DDPSM', 'mimic', 1e-4, 15, 5)

    # %%
    cis_list, brs_list , roc_aoc_list = result(args.n_run, args.model, args.dataset , args.lr, args.k1, args.k2, args.epoch, args.eta, args.edit_censor, args.censor_rate)

    # %%
    # support_cis_list, support_brs_list , support_roc_aoc_list = result(5, 'DDPSM', 'mimic', 1e-4, 9, 15)



