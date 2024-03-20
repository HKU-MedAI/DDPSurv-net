from auton_survival import datasets

outcomes, features = datasets.load_support()

# %%

from auton_survival.preprocessing import Preprocessor

cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp',
             'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph',
             'glucose', 'bun', 'urine', 'adlp', 'adls']

features = Preprocessor().fit_transform(features, cat_feats=cat_feats, num_feats=num_feats)

import numpy as np

horizons = [0.25, 0.5, 0.75]
times = np.quantile(outcomes.time[outcomes.event == 1], horizons).tolist()

x, t, e = features.values.astype(float), outcomes.time.values.astype(float), outcomes.event.values.astype(float)

n = len(x)

tr_size = int(n * 0.70)
vl_size = int(n * 0.10)
te_size = int(n * 0.20)

print(tr_size, vl_size, te_size)

x_train, x_test, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size + vl_size]
t_train, t_test, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size + vl_size]
e_train, e_test, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size + vl_size]

models = []

from auton_survival.models.dpsm import DeepDP
from auton_survival.models.dsm import DeepSurvivalMachines

# model = DeepDP(k=3,
#                distribution='LogNormal',
#                layers=[100])
model = DeepSurvivalMachines(
    k=1,
    distribution="LogNormal",
    layers=[100]
)
# The fit method is called to train the model
model.fit(x_train, t_train, e_train, iters=100, learning_rate=0.001)

models.append([[model.compute_nll(x_val, t_val, e_val), model]])

best_model = min(models)
model = best_model[0][1]

out_risk = model.predict_risk(x_test, times)
out_survival = model.predict_survival(x_test, times)

from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc

cis = []
brs = []

et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                 dtype = [('e', bool), ('t', float)])
et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                 dtype = [('e', bool), ('t', float)])
et_val = np.array([(e_val[i], t_val[i]) for i in range(len(e_val))],
                 dtype = [('e', bool), ('t', float)])

for i, _ in enumerate(times):
    cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])
brs.append(brier_score(et_train, et_test, out_survival, times)[1])
roc_auc = []
for i, _ in enumerate(times):
    roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0])
for horizon in enumerate(horizons):
    print(f"For {horizon[1]} quantile,")
    print("TD Concordance Index:", cis[horizon[0]])
    print("Brier Score:", brs[0][horizon[0]])
    print("ROC AUC ", roc_auc[horizon[0]][0], "\n")

# %%

'''                                                                                                                                   | 16/100 [00:05<00:26,  3.16it/s]
For 0.25 quantile,
TD Concordance Index: 0.7586835202988933
Brier Score: 0.11128451332353094
ROC AUC  0.7662427960769529 

For 0.5 quantile,
TD Concordance Index: 0.7001363224012463
Brier Score: 0.18338282896942423
ROC AUC  0.7211280437788559 

For 0.75 quantile,
TD Concordance Index: 0.6533129637564933
Brier Score: 0.22365597413329832
ROC AUC  0.7059332404434483 
'''

'''
For 0.25 quantile,
TD Concordance Index: 0.7588144276400827
Brier Score: 0.1112550550087257
ROC AUC  0.7664117240940316 

For 0.5 quantile,
TD Concordance Index: 0.7002115092590766
Brier Score: 0.18335736944679448
ROC AUC  0.721188099088331 

For 0.75 quantile,
TD Concordance Index: 0.653207107446695
Brier Score: 0.22364208736179372
ROC AUC  0.7057830011510269 
'''


'''
For 0.25 quantile,
TD Concordance Index: 0.7637676783877877
Brier Score: 0.1110828884326944
ROC AUC  0.7713865317853978 

For 0.5 quantile,
TD Concordance Index: 0.7011815429822312
Brier Score: 0.18326757245019656
ROC AUC  0.7222500527314912 

For 0.75 quantile,
TD Concordance Index: 0.653984092760614
Brier Score: 0.2238907197929432
ROC AUC  0.707597988732053 
'''