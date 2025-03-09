# write a python file to generate synthetic data 
# generate convariate 
# generate time t conditional on covariate
# generate delta

import numpy as np
from scipy.stats import bernoulli
from sklearn.model_selection import train_test_split

def edit_censor(event, time):
    assert event.shape[0] == time.shape[0]
    for i in range(event.shape[0]):
        if event[i] == 0:
            time[i] = np.random.uniform(0, time[i])
    return time



    
n = 10000
censor_rate = 0.75
event = bernoulli.rvs(censor_rate, size=n)
# generate covariate, shape = (n, 8)
covariate_continous = np.random.normal(0, 1, (n, 7))
covariate_categorical = np.random.randint(0, 8, (n, 1))
print(covariate_categorical)
covariate = np.concatenate((covariate_continous, covariate_categorical), axis=1)

# generate time, shape = (n,)
beta_mu = np.random.normal(-2, 2, 8)
beta_sigma = np.random.normal(-1, 1, 8)
shape = np.abs(np.dot(covariate, beta_mu))
scale = np.abs(np.dot(covariate, beta_sigma))
shape = shape / shape.max()
scale = scale / scale.max()
print(shape[0:10], scale[0:10])
time = np.zeros(n)
for i in range(n):
    time[i] = np.random.lognormal(shape[i], scale[i])

time = edit_censor(event, time)

# train test split

x_train, x_test, t_train, t_test, e_train, e_test = train_test_split(covariate, time, event, test_size=0.2, random_state=42)

# save data

np.save('datasets/new/synthetic_x_train_new.npy', x_train)
np.save('datasets/new/synthetic_x_test_new.npy', x_test)
np.save('datasets/new/synthetic_t_train_new.npy', t_train)
np.save('datasets/new/synthetic_t_test_new.npy', t_test)
np.save('datasets/new/synthetic_e_train_new.npy', e_train)
np.save('datasets/new/synthetic_e_test_new.npy', e_test)