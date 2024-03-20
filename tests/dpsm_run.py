import pandas as pd
import sys

# import os
# os.chdir('/home/r10user10/Documents/Jiacheng/dspm-auton-survival-main/')
# sys.path.append('/home/r10user10/Documents/Jiacheng/dspm-auton-survival-main/')
# print(os.getcwd())

from auton_survival.datasets import load_dataset

# Load the SUPPORT dataset
outcomes, features = load_dataset(dataset='SUPPORT')

# Identify categorical (cat_feats) and continuous (num_feats) features
cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
             'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
             'glucose', 'bun', 'urine', 'adlp', 'adls']


import numpy as np
from sklearn.model_selection import train_test_split

# Split the SUPPORT data into training, validation, and test data
x_tr, x_te, y_tr, y_te = train_test_split(features, outcomes, test_size=0.2, random_state=1)
x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.25, random_state=1) 

print(f'Number of training data points: {len(x_tr)}')
print(f'Number of validation data points: {len(x_val)}')
print(f'Number of test data points: {len(x_te)}')


from auton_survival.preprocessing import Preprocessor

# Fit the imputer and scaler to the training data and transform the training, validation and test data
preprocessor = Preprocessor(cat_feat_strat='ignore', num_feat_strat= 'mean') 
transformer = preprocessor.fit(features, cat_feats=cat_feats, num_feats=num_feats,
                                one_hot=True, fill_value=-1)
x_tr = transformer.transform(x_tr)
x_val = transformer.transform(x_val)
x_te = transformer.transform(x_te)

import numpy as np
from auton_survival.estimators import SurvivalModel
from auton_survival.metrics import survival_regression_metric
from sklearn.model_selection import ParameterGrid

# Define parameters for tuning the model
param_grid = {'layers' : [[100], [100, 100], [200]],
              'distribution' : ['Weibull', 'LogNormal'],
              'max_features' : ['sqrt', 'log2']
             }

params = ParameterGrid(param_grid)

# Define the times for tuning the model hyperparameters and for evaluating the model
times = np.quantile(y_tr['time'][y_tr['event']==1], np.linspace(0.25, 1, 4)).tolist()

# Perform hyperparameter tuning 
models = []
for param in params:
    model = SurvivalModel('dpsm', random_seed=0, layers=param['layers'], distribution=param['distribution'], max_features=param['max_features'])
    
    # The fit method is called to train the model
    model.fit(x_tr, y_tr)

    # Obtain survival probabilities for validation set and compute the Integrated Brier Score 
    predictions_val = model.predict_survival(x_val, times)
    print(predictions_val.shape)
    metric_val = survival_regression_metric('ibs', y_val, predictions_val, times, y_tr)
    models.append([metric_val, model])
   
    
# Select the best model based on the mean metric value computed for the validation set
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

plot_performance_metrics(results, times)
