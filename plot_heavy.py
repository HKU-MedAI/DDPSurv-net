import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# path = 'k2_0.csv'
# df = pd.read_csv(path)
# with_index = [0,3,6]
# without_index = [1,4,7]
# with_heavy = df['0.25'][with_index].values
# without_heavy = df['0.25'][without_index].values
# diff = with_heavy - without_heavy
with_heavy_75 = [0.65960929, 0.66324755 ,0.68419484, 0.68419485]
without_heavy_75 = [0.65725505,0.50690384 ,0.66350353, 0.65708504]
with_heavy_90 = [0.65700223, 0.65791073, 0.66303348, 0.66770752]
without_heavy_90 = [0.44048509, 0.50339398 , 0.64382219, 0.64659713]
d = 0.3
x = np.arange(4)
x_axis = ['Support', 'Synthetic', 'MIMIC-III', 'MIMIC-IV']
plt.figure(figsize=(10,6))
plt.ylim(0,1)
plt.bar(x, with_heavy_90, width=d, color='orange', label='With Heavy Tails')
plt.bar(x + d, without_heavy_90, width=d, color='dodgerblue', label='Without Heavy Tails')
plt.xticks(x + d/2, x_axis, size=20, fontweight='bold')
plt.yticks(size=20, fontweight='bold')
plt.xlabel('Datasets', size=20, fontweight='bold')
plt.ylabel('C-Index', size=20, fontweight='bold')
plt.legend(loc="upper right", ncol=2)
plt.title('Heavy-tail distribution effects for Horizon 0.90',size=20, pad=30, fontweight='bold')
plt.tight_layout()
plt.show()
plt.savefig('heavy_tail_effects_90.png')