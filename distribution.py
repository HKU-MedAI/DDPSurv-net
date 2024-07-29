from scipy.stats import lognorm, weibull_min, cauchy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def logcauchy(x, mu, sigma):
    pdf = np.exp(-np.log(x * np.pi) - np.log(sigma) - np.log(1 + ((np.log(x) - mu) / sigma) ** 2))
    return pdf

def lognormal(x, mu, sigma):
    return lognorm.pdf(x, loc=mu, scale=sigma)

def weibull(x, mu, sigma):
    pdf = mu/sigma * (x/sigma)**(mu-1) * np.exp(-(x/sigma)**mu)
    return pdf

if __name__ == '__main__':

    pass