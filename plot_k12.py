import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = 'ablation_study/mimic4/mimic4_cis_mean_dict.npy'
cis = np.load(path, allow_pickle=True).item()
np_cis = np.array(list(cis.values()))
print(np_cis.shape)
k2_index = np.arange(0,400,20)
# k1_index = [6, 8, 10, 12, 14, 16, 18]
k1_index = [6, 8, 10, 12, 14, 16]
# k1_index = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# k1_index = [3, 6, 9, 12, 15, 18]
# cis_1 = np_cis[0:20,0][k2_index]
# cis_4 = np_cis[60:80,0][k2_index]
# cis_7 = np_cis[120:140,0][k2_index]
# cis_10 = np_cis[180:200,0][k2_index]
# cis_13 = np_cis[240:260,0][k2_index]
# cis_16 = np_cis[300:320,0][k2_index]
# cis_19 = np_cis[360:380,0][k2_index]
cis_1 = np_cis[:,0][k2_index + 1][k1_index]
cis_2 = np_cis[:,0][k2_index + 2][k1_index]
cis_3 = np_cis[:,0][k2_index + 3][k1_index]
cis_4 = np_cis[:,0][k2_index + 4][k1_index]
cis_5 = np_cis[:,0][k2_index + 5][k1_index]
cis_6 = np_cis[:,0][k2_index + 6][k1_index]
cis_7 = np_cis[:,0][k2_index + 7][k1_index]
cis_8 = np_cis[:,0][k2_index + 8][k1_index]
cis_9 = np_cis[:,0][k2_index + 9][k1_index]
cis_10 = np_cis[:,0][k2_index + 10][k1_index]
cis_11 = np_cis[:,0][k2_index + 11][k1_index]
cis_12 = np_cis[:,0][k2_index + 12][k1_index]
cis_13 = np_cis[:,0][k2_index + 13][k1_index]
cis_14 = np_cis[:,0][k2_index + 14][k1_index]
cis_15 = np_cis[:,0][k2_index + 15][k1_index]
cis_16 = np_cis[:,0][k2_index + 16][k1_index]
cis_17 = np_cis[:,0][k2_index + 17][k1_index]
cis_18 = np_cis[:,0][k2_index + 18][k1_index]
cis_19 = np_cis[:,0][k2_index + 19][k1_index]

plt.figure(figsize=(10, 6))
plt.ylim(0.625, 0.825)
plt.plot(cis_6, label = 'k2 = 6', linewidth =3)
plt.plot(cis_7, label = 'k2 = 7', linewidth =3)
plt.plot(cis_16, label = 'k2 = 16', linewidth =3)
plt.plot(cis_17, label = 'k2 = 17', linewidth =3)

plt.xticks(range(len(k1_index)), k1_index, size=20, fontweight='bold')
plt.yticks(size=20, fontweight='bold')
plt.xlabel('k1', size=20, fontweight='bold')
plt.ylabel('C-Index', size=20, fontweight='bold')
plt.title('C-index (0.25 Horizon) under different k1 & k2',size=20, pad=30, fontweight='bold')
plt.legend(ncol=2, loc="upper right", fontsize=15)
plt.tight_layout()
plt.show()
plt.savefig('k12.png')