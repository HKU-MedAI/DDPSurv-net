

import pandas as pd
import sys

import os
os.chdir('/home/r10user10/Documents/Jiacheng/dspm-auton-survival-main/')
sys.path.append('/home/r10user10/Documents/Jiacheng/dspm-auton-survival-main')

print(os.getcwd())
print(sys.path)

from auton_survival.datasets import load_dataset

# Load the SUPPORT dataset
outcomes, features = load_dataset(dataset='SUPPORT')

from auton_survival.preprocessing import Preprocessor
cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
	     'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
             'glucose', 'bun', 'urine', 'adlp', 'adls']

features = Preprocessor().fit_transform(features, cat_feats=cat_feats, num_feats=num_feats)


import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
horizons = [0.25, 0.5, 0.75]
times = np.quantile(outcomes.time[outcomes.event==1], horizons).tolist()

x, t, e = features.values, outcomes.time.values, outcomes.event.values

n = len(x)

tr_size = int(n*0.70)
vl_size = int(n*0.10)
te_size = int(n*0.20)

x_train, x_test, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size+vl_size]
t_train, t_test, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size+vl_size]
e_train, e_test, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size+vl_size]

import numpy as np
from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric
from sklearn.model_selection import ParameterGrid
from auton_survival.models.dpsm import DeepDP

# Define parameters for tuning the model
param_grid = {'k' : [4],
              'distribution' : ['LogNormal', 'Weibull'],
              'learning_rate' : [ 1e-4, 1e-3],
              'layers' : [ [], [100], [100, 100] ]
             }
params = ParameterGrid(param_grid)


# Perform hyperparameter tuning 
models = []
for param in params:
    model = DeepDP(k = param['k'],
                                 distribution = param['distribution'],
                                 layers = param['layers'])
    
    # The fit method is called to train the model
    model.fit(x_train, t_train, e_train, iters = 100, learning_rate = param['learning_rate'])

    # Obtain survival probabilities for validation set and compute the Integrated Brier Score 
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
    
    
'''# Select the best model based on the mean metric value computed for the validation set
metric_vals = [i[0] for i in models]
first_min_idx = metric_vals.index(min(metric_vals))
model = models[first_min_idx][1]

print(metric_vals)
print(first_min_idx)



import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

def plot_performance_metrics(results, times):
  """Plot Brier Score, ROC-AUC, and time-dependent concordance index
  for survival model evaluation.

  Parameters
  -----------
  results : dict
      Python dict with key as the evaulation metric
  times : float or list
      A float or list of the times at which to compute
      the survival probability.

  Returns
  -----------
  matplotlib subplots

  """

  colors = ['blue', 'green']
  gs = gridspec.GridSpec(len(results), 1, wspace=0.3)
  plt.figure(figsize=(6,8))
  for fi, result in enumerate(results.keys()):
    val = results[result]
    x = [str(round(t, 1)) for t in times]
    ax = plt.subplot(gs[fi, 0]) # row 0, col 0
    ax.set_xlabel('Time')
    ax.set_ylabel(result)
    ax.set_ylim(0, 1)
    ax.bar(x, val, color=colors[fi])
    plt.xticks(rotation=0)
    plt.savefig('./dpsm_result.jpg')
  plt.show()




# Obtain survival probabilities for test set
predictions_te = model.predict_survival(x_te, times)

# Compute the Brier Score and time-dependent concordance index for the test set to assess model performance
results = dict()
results['Brier Score'] = survival_regression_metric('brs', outcomes=y_te, predictions=predictions_te, 
                                                    times=times, outcomes_train=y_tr)
results['Concordance Index'] = survival_regression_metric('ctd', outcomes=y_te, predictions=predictions_te, 
                                                    times=times, outcomes_train=y_tr)
print(results)

plot_performance_metrics(results, times)'''
