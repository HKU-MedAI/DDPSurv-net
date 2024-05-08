import sys
import os
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from auton_survival.models.dsm import DeepSurvivalMachines
from auton_survival.models.dpsm import DeepDP
from auton_survival import datasets
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from pycox.datasets import from_kkbox
# test

kkbox_data = from_kkbox._DatasetKKBoxChurn()
# kkbox_data.download_kkbox()

df = kkbox_data.read_df()


e = np.array(df.event)
t = np.array(df.duration)
x = df.drop(columns=['event', 'duration', 'msno'])


le = LabelEncoder()
x['gender'] = le.fit_transform(x['gender'])
x['registered_via'] = le.fit_transform(x['registered_via'])
x['city'] = le.fit_transform(x['city'])
x = np.array(x).astype(float)


# path = '/home/r10user10/Documents/Jiacheng/dspm-auton-survival'
# os.chdir(path)


n = len(x)

tr_size = int(n * 0.70)
vl_size = int(n * 0.10)
te_size = int(n * 0.20)

print(tr_size, vl_size, te_size)

x_train, x_test, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size + vl_size]
t_train, t_test, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size + vl_size]
e_train, e_test, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size + vl_size]

models = []


# model = DeepDP(k=3,
#                distribution='LogNormal',
#                layers=[100])
model = DeepSurvivalMachines(
    k=4,
    distribution="LogNormal",
    # distribution="Weibull",
    layers=[100]
)
# The fit method is called to train the model
<< << << < Updated upstream
model.fit(x_train, t_train, e_train, iters=100, learning_rate=0.000001)
== == == =
model.fit(x_train, t_train, e_train, iters=1, learning_rate=0.0001)
>>>>>> > Stashed changes

trained_weights = model.trained_weights
print(np.isnan(trained_weights).sum())
# import ipdb
# ipdb.set_trace()

horizons = [0.25, 0.5, 0.75]
times = np.quantile(t[e == 1], horizons).tolist()
out_risk = model.predict_risk(x_test, times)
out_survival = model.predict_survival(x_test, times)


cis = []
brs = []

et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                    dtype=[('e', bool), ('t', float)])
et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                   dtype=[('e', bool), ('t', float)])
et_val = np.array([(e_val[i], t_val[i]) for i in range(len(e_val))],
                  dtype=[('e', bool), ('t', float)])

for i, _ in enumerate(times):
    cis.append(concordance_index_ipcw(
        et_train, et_test, out_risk[:, i], times[i])[0])

brs.append(brier_score(et_train, et_test, out_survival, times)[1])
roc_auc = []
for i, _ in enumerate(times):
    roc_auc.append(cumulative_dynamic_auc(
        et_train, et_test, out_risk[:, i], times[i])[0])
for horizon in enumerate(horizons):
    print(f"For {horizon[1]} quantile")
    print("TD Concordance Index:", cis[horizon[0]])
    print("Brier Score:", brs[0][horizon[0]])
    print("ROC AUC ", roc_auc[horizon[0]][0], "\n")


# import seaborn as sns
# import matplotlib.pyplot as plt

<< << << < Updated upstream
figs, axes = plt.subplots(1, 3, figsize=(15, 5))
iter_idx = [97, 98, 99]

for idx in range(3):
    sns.kdeplot(trained_weights[iter_idx[idx]][:, 0], fill=True, ax=axes[idx])
    sns.kdeplot(trained_weights[iter_idx[idx]][:, 1], fill=True, ax=axes[idx])
    sns.kdeplot(trained_weights[iter_idx[idx]][:, 2], fill=True, ax=axes[idx])
    sns.kdeplot(trained_weights[iter_idx[idx]][:, 3], fill=True, ax=axes[idx])
    axes[idx].set_title(f'Iter {iter_idx[idx]}')
    axes[idx].set_xlim(0.15, 0.4)

plt.savefig("kkbox.png")
== == == =
# figs, axes = plt.subplots(1, 3, figsize=(15, 5))
# iter_idx = [24, 25, 26]

# for idx in range(3):
#     sns.kdeplot(trained_weights[iter_idx[idx]][:, 0], fill=True, ax=axes[idx])
#     sns.kdeplot(trained_weights[iter_idx[idx]][:, 1], fill=True, ax=axes[idx])
#     sns.kdeplot(trained_weights[iter_idx[idx]][:, 2], fill=True, ax=axes[idx])
#     sns.kdeplot(trained_weights[iter_idx[idx]][:, 3], fill=True, ax=axes[idx])
#     axes[idx].set_title(f'Iter {iter_idx[idx]}')
#     axes[idx].set_xlim(0.249, 0.251)
>>>>>> > Stashed changes
