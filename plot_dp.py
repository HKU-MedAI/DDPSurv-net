import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DDPSurv_df = pd.read_csv('DDPSurv.csv')
DSM_df = pd.read_csv('DSM.csv')

np_DDPSurv = np.array(DDPSurv_df)[:,1:]
np_DSM = np.array(DSM_df)[:,1:]
diff = np_DDPSurv - np_DSM

x_axis = list(DDPSurv_df.columns[1:])
x = np.arange(len(x_axis))
d = 0.2
colors=['orange','dodgerblue','grey']
horizon = [0.25, 0.5, 0.75, 0.9]
plt.figure(figsize=(10, 7))
plt.ylim(0.65,0.8)
plt.bar(x, list(np.mean(np_DDPSurv, axis=0)), width = d, color=colors[0], label='DDPSurv')
plt.bar(x + d, list(np.mean(np_DSM, axis=0)), width = d, color=colors[1], label='DSM')
# plt.bar(x + 2 * d, list(np.mean(diff, axis=0)), width = d, color=colors[2], label='Difference')
plt.xticks(x + d/2, x_axis, size=20, fontweight='bold')
plt.yticks(size=20, fontweight='bold')
plt.xlabel('Censor rate', size=20, fontweight='bold')
plt.ylabel('C-Index', size=20, fontweight='bold')
plt.margins(0.05)
plt.legend(loc="upper right", fontsize=15) 
plt.title(f'Effects of Using Dirichlet Process ',size=20, pad=30, fontweight='bold')
plt.tight_layout()
plt.show()
plt.savefig(f'dp.png')
plt.close()