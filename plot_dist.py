import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import argparse
import os


colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 
          'b', 'g', 'r', 'c', 'm', 'y', 'k']

path = './results/mimic4/LogNormal_10_2_distribution'
if not os.path.exists(path):
    os.makedirs(path)
df = pd.read_csv(path+'.csv')
for i in range(1, len(df.columns)):
    plt.ylim(0, 0.0025)  
    x = df.iloc[:,0]
    y = df.iloc[:,i]
    plt.axis('off')
    plt.plot(x, y, label=f'Component {i}', color=colors[i-1], linewidth=20)
    a = plt.gca()
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)
    plt.show()
    plt.savefig(path + f'/component_{i}.png')
    # close the plt 
    plt.close()

