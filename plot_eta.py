import matplotlib.pyplot as plt
import numpy as np

path = 'ablation_study/mimic3/2_12_eta_cis_dict.npy'
cis_eta = np.load(path, allow_pickle=True).item()
np_eta = np.array(list(cis_eta.values()))

plt.figure(figsize=(8, 6))
plt.ylim(0.625, 0.9)
plt.plot(range(6), np_eta[:,0], label='Horizon 0.25', linewidth=5)
plt.plot(range(6), np_eta[:,1], label='Horizon 0.5', linewidth=5)
plt.plot(range(6), np_eta[:,2], label='Horizon 0.75', linewidth=5)
plt.plot(range(6), np_eta[:,3], label='Horizon 0.9', linewidth=5)
plt.xticks(range(6), list(cis_eta.keys()), size=20, fontweight='bold')
plt.yticks(size=20, fontweight='bold')
plt.xlabel('eta', size=20, fontweight='bold')
plt.ylabel('C-Index', size=20, fontweight='bold')
plt.title('C-index under different eta',size=20, pad=30, fontweight='bold')
plt.legend(ncol=2, loc="upper right", fontsize=15)
plt.tight_layout()
plt.show()
plt.savefig('eta_cis.png')