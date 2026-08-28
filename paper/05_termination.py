# %%
import numpy as np
from sys import path
from pathlib import Path
# Sets parent directory as a string path
parent_dir = str(Path(__file__).resolve().parent.parent)
path.append(parent_dir)
# path.append('/home/peter.barkley/code/parallel_sdp/code')
from snl import generateRandomData, solve_snl_fusion, getZfromNeighbors, getSNLProxData, node_objective_value
from oars import solve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from datetime import datetime
import csv
import os
from functools import partial
# import matplotlib
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['text.usetex'] = True

# %%
n = 30
m = 6
d = 2

# %%
itrs = 800
numtests = 300
cvx_devs = []
mpps_devs = []

data_dict = {}

class callback:
    def __init__(self, n, d, dx, a, aa, Ni, Na):
        self.window = 100
        self.var = np.inf
        self.varidx = 0
        self.obj = np.inf
        self.objidx = 0
        self.idx = 0
        self.n = n
        self.d = d
        self.dx = dx
        self.a = a
        self.aa = aa
        self.Ni = Ni
        self.Na = Na


    def __call__(self, itr, all_x, **kwargs):
        self.checkObjective(all_x, itr)
        if self.objidx + self.window < itr:
            return True


    def checkObjective(self, X, itr):
        val = 0
        for i in range(self.n):
            val += node_objective_value(X[i+self.n], self.dx, i, self.Ni[i], self.Na[i], self.d, self.a, self.aa)
        if val < self.obj:
            self.obj = val
            self.objidx = itr

X_oars_dist = np.zeros((n,d))
s = 0
oarsalpha = 10.0
rand_uniform = partial(np.random.uniform, -1.0, 1.0)
rand_uniform.__name__ = 'uniform'
print('\t'.join(['date\t', 'time\t', 'nf', 'rand', 's', 'cvx_mse', 'mpps_mse', 'dist_mpps_mse', 'itrs']))
for nf in [.03, .05, .07]:
    for rand in [np.random.randn, rand_uniform]:
        for i in range(numtests):
            print(datetime.now(), nf, rand.__name__, s, sep='\t', end='\t')
            a, x, da, dx, aa, Ni, Na = generateRandomData(n, m, d, rd=.7, nf=nf, seed=s, rand=rand) 
            center = np.mean(a, axis=0)
            xnorm = np.linalg.norm(x)

            # Reference solution
            X_cvx, cvx_val = solve_snl_fusion(a, n, aa, dx, Ni, Na)
            cvx_mse = np.mean([np.linalg.norm(X_cvx[d+i, :d] - x[i])**2 for i in range(n)])
            print(f'{cvx_mse:.3e}', end='\t')

            # OARS
            Z = getZfromNeighbors(Ni)
            data, proxlist = getSNLProxData(n, a, d, dx, aa, Ni, Na)
            primal = np.zeros((n+d, n+d))
            primal[:d, :d] = np.eye(d)
            cb = callback(n, d, dx, a, aa, Ni, Na)
            X_oars, cleanlog, _ = solve(2*n, data, proxlist, W=Z, Z=Z, warmstartprimal=primal, alpha=oarsalpha, gamma=1.0, itrs=itrs, verbose=False, callback=cb)
            mpps_mse = np.mean([np.linalg.norm(X_oars[i+d, :d] - x[i])**2 for i in range(n)])
            # itr_stop = len(cleanlog[0])
            for i in range(n):
                X_oars_dist[i] = proxlist[i+n].log[-1][-d:]
            mpps_mse_dist = np.mean([np.linalg.norm(X_oars_dist[i] - x[i])**2 for i in range(n)])
            print(f'{mpps_mse:.3e}\t{mpps_mse_dist:.3e}\t{len(proxlist[0].log)}')
            data_dict[(nf, rand.__name__, s)] = (xnorm, cvx_mse, mpps_mse, mpps_mse_dist)
            s += 1


def write(filename, data_dict):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['nf', 'rand', 's', 'xnorm', 'IP_MSE', 'MPPS_MSE', 'MPPS_MSE_Dist'])
        for s, values in data_dict.items():
            writer.writerow(list(s) + list(values))

 
# Get the directory of the current script
dir_path = os.path.dirname(os.path.abspath(__file__))

# Create the full path to the file
file_path = os.path.join(dir_path, "term_output.csv")
write(file_path, data_dict)



plt.figure(figsize=(10, 10))
plt.hist([mpps_devs_i - cvx_devs_i for mpps_devs_i, cvx_devs_i in zip(mpps_devs, cvx_devs)], bins=np.arange(-0.16, 0.1, 0.02))
plt.xlabel(r'$\frac{1}{n}\left(\|\hat{X} - X_0\|_F^2 - \|\bar{X} - X_0\|_F^2 \right)$')
plt.ylabel('Count')
# vertical line at 0 in red
plt.axvline(x=0, color='r', linestyle='--')
# tight layout
plt.tight_layout()
plt.savefig('figs/fig_4_paired_diffs_mean.pdf')