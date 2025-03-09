import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

csv_path = 'ablation_study/support/2_10_censor.csv'
df = pd.read_csv(csv_path)
censor_rate = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 'default']
DDPSM = list(df.iloc[0].values[1:])
DCM = list(df.iloc[2].values[1:])
diff = list(df.iloc[0].values[1:] - df.iloc[2].values[1:])
x_axis = list(censor_rate)
x_axis = [str(j) for j in x_axis]
x = np.arange(len(x_axis))
d = 0.2
sns.set_theme(style='ticks', context='paper')
plt.figure(figsize=(10, 8))
plt.ylim(0,1)
plt.bar(x, DDPSM, width = d, label='DDPSurv')
plt.bar(x + d, DCM, width = d,  label='DeepCoxMixture')
plt.bar(x + 2 * d, diff, width = d, label='Difference')
plt.xticks(x + d, x_axis, size=20)
plt.yticks(size=20)
plt.xlabel('Censor rate', size=20)
plt.ylabel('C-Index', size=20)
plt.margins(0.05)
plt.legend(loc="upper right", fontsize=15) 
plt.title(f'C-index under different censor rates',size=20, pad=30)
plt.tight_layout()
plt.show()
plt.savefig(f'ablation_study/support/censor_cis_mean.png')
plt.close()

