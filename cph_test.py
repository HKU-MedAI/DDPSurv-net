from lifelines import CoxPHFitter
import numpy as np
from dspm_dataset import support, synthetic, kkbox, mimic


def cph(dataset):
    if dataset == 'support':
        x_train, t_train , e_train, x_test, t_test , e_test = support()
        print(x_train.shape[-1])
    if dataset == 'synthetic':
        x_train, t_train , e_train, x_test, t_test , e_test = synthetic()
        print(x_train.shape[-1])
    if dataset == 'kkbox':
        x_train, t_train , e_train, x_test, t_test , e_test = kkbox()
        print(x_train.shape[-1])
    if dataset == 'mimic':
        x_train, t_train , e_train, x_test, t_test , e_test = mimic()  
        print(x_train.shape[-1])
    
    cph = CoxPHFitter(penalizer=0.1)
    # x_train = np.concatenate((x_train, np.expand_dims(e_train,axis=1)), axis=-1)
    # x_test  = np.concatenate((x_test, np.expand_dims(e_test,axis=1)), axis=-1)
    import pandas as pd
    data_train = np.concatenate((x_train, np.expand_dims(t_train,axis=1), np.expand_dims(e_train,axis=1)), axis=-1)
    # print(pd.DataFrame(data_train).head(5))
    data_test = np.concatenate((x_test, np.expand_dims(t_test,axis=1), np.expand_dims(e_test,axis=1)), axis=-1)
    cph.fit(pd.DataFrame(data_train), duration_col=x_train.shape[-1], event_col=x_train.shape[-1]+1)
    
    horizons = [0.25, 0.5, 0.75, 0.9]
    x = np.concatenate((x_train, x_test), axis=0)
    t = np.concatenate((t_train, t_test), axis=0)
    e = np.concatenate((e_train, e_test), axis=0)
    times = np.quantile(t[e==1], horizons).tolist()


    # print(x_test.shape)
    cph_prediction = np.array(cph.predict_survival_function(pd.DataFrame(x_test))).T
    # print(cph_prediction.shape)
    out_survival = np.quantile(cph_prediction, horizons, axis=-1).T
    # print(out_survival[0])
    # print(out_survival[1000])
    out_risk = 1 - out_survival
    

    from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc

    cis = []
    brs = []

    et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                    dtype = [('e', bool), ('t', float)])

    et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                    dtype = [('e', bool), ('t', float)])

    for i, _ in enumerate(times):
        cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])
    brs.append(brier_score(et_train, et_test, out_survival, times)[1])
    roc_auc = []
    for i, _ in enumerate(times):
        roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0])
    for horizon in enumerate(horizons):
        print(f"For {horizon[1]} quantile")
        print("TD Concordance Index:", cis[horizon[0]])
        print("Brier Score:", brs[0][horizon[0]])
        print("ROC AUC ", roc_auc[horizon[0]][0], "\n")
    return cis, brs, roc_auc
    
cph('mimic')


