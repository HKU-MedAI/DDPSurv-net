import numpy as np


from auton_survival.models.dpsm import DeepDP


def kkbox():
    from pycox.datasets import from_kkbox


    kkbox_data = from_kkbox._DatasetKKBoxChurn()
    # kkbox_data.download_kkbox()

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

    n = len(x)

    tr_size = int(n * 0.80)
    te_size = int(n * 0.20)


    x_train, x_test = x[:tr_size], x[-te_size:]
    t_train, t_test = t[:tr_size], t[-te_size:]
    e_train, e_test = e[:tr_size], e[-te_size:]
    return x_train, t_train , e_train, x_test, t_test , e_test


x_train, t_train , e_train, x_test, t_test , e_test = kkbox()


lr = 0.001

model = DeepDP(k=10,
               distribution='LogNormal',
               layers=[100,100])
# The fit method is called to train the model
model.fit(x_train, t_train, e_train, iters=100, learning_rate=lr)
horizons = [0.25, 0.5, 0.75, 0.9]
x = np.concatenate((x_train, x_test), axis=0)
t = np.concatenate((t_train, t_test), axis=0)
e = np.concatenate((e_train, e_test), axis=0)
times = np.quantile(t[e == 1], horizons).tolist()
out_risk = 1 - model.predict_survival(x_test, times)
out_survival = model.predict_survival(x_test, times)

from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc

cis = []
brs = []

et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                    dtype=[('e', bool), ('t', float)])
# print(et_train)
et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                   dtype=[('e', bool), ('t', float)])
# et_val = np.array([(e_val[i], t_val[i]) for i in range(len(e_val))],
#                  dtype = [('e', bool), ('t', float)])
# print(et_train[0:10])
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