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

x_train, x_test, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size + vl_size]
t_train, t_test, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size + vl_size]
e_train, e_test, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size + vl_size]

models = []

from auton_survival.models.dpsm import DeepDP

model = DeepDP(k=3,
               distribution='LogNormal',
               layers=[100])
# The fit method is called to train the model
model.fit(x_train, t_train, e_train, iters=100, learning_rate=0.001)
models.append([[model.compute_nll(x_val, t_val, e_val), model]])

best_model = min(models)
model = best_model[0][1]
