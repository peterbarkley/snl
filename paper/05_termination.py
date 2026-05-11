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


    def __call__(self, itr, x, v, **kwargs):
        self.checkObjective(x, itr)
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
print('\t'.join(['date\t', 'time', 'nf', 'rand', 's', 'cvx_dev', 'mpps_dev', 'distributed_mpps_dev', 'itrs']))
for nf in [.03, .05, .07]:
    for rand in [np.random.randn, rand_uniform]:
        for i in range(numtests):
            print(datetime.now(), nf, rand.__name__, s, end='\t')
            a, x, da, dx, aa, Ni, Na = generateRandomData(n, m, d, rd=.7, nf=nf, seed=s, rand=rand) #n, m, d=2, rd=1, nf=0, cutoff=7, seed=0, rand=np.random.randn
            # x_vals.append(x)
            center = np.mean(a, axis=0)
            xnorm = np.linalg.norm(x)
            # xnorms.append(xnorm)

            # Reference solution
            X_cvx, cvx_val = solve_snl_fusion(a, n, aa, dx, Ni, Na)
            # X_cvx, cvx_val = solve_snl_vec(a, n, aa, dx, Ni, Na) # CVXPY (slower, especially as size increases)
            cvx_dev = np.linalg.norm(X_cvx[d:, :d] - x, 'fro')
            cvx_devs.append(cvx_dev)
            cvx_mean_dev = np.mean([np.linalg.norm(X_cvx[d+i, :d] - x[i]) for i in range(n)])
            print(f'{cvx_dev:.3f}', end='\t')

            # OARS
            Z = getZfromNeighbors(Ni)
            data, proxlist = getSNLProxData(n, a, d, dx, aa, Ni, Na)
            primal = np.zeros((n+d, n+d))
            primal[:d, :d] = np.eye(d)
            cb = callback(n, d, dx, a, aa, Ni, Na)
            X_oars, cleanlog, _, _ = solve(2*n, data, proxlist, W=Z, Z=Z, warmstartprimal=primal, alpha=oarsalpha, gamma=1.0, itrs=itrs, verbose=False, callback=cb)
            mpps_dev = np.linalg.norm(X_oars[d:, :d] - x, 'fro')
            mpps_devs.append(mpps_dev)
            mpps_mean_dev = np.mean([np.linalg.norm(X_oars[i+d, :d] - x[i]) for i in range(n)])
            itr_stop = len(cleanlog[0])
            for i in range(n):
                X_oars_dist[i] = cleanlog[i+n][itr_stop-1][-d:]
            mpps_dev_dist = np.linalg.norm(X_oars_dist - x, 'fro')
            mpps_mean_dev_dist = np.mean([np.linalg.norm(X_oars_dist[i] - x[i]) for i in range(n)])
            print(f'{mpps_dev:.3f} {mpps_dev_dist:.3f} {len(cleanlog[0])}')
            data_dict[(nf, rand.__name__, s)] = (xnorm, cvx_dev, cvx_mean_dev, mpps_dev, mpps_mean_dev, mpps_dev_dist, mpps_mean_dev_dist)
            s += 1


def write(filename, data_dict):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['nf', 'rand', 's', 'xnorm', 'IP_Deviation', 'IP_Mean_Deviation', 'MPPS_Deviation', 'MPPS_Mean_Deviation', 'MPPS_Deviation_Dist', 'MPPS_Mean_Deviation_Dist'])
        for s, values in data_dict.items():
            writer.writerow(list(s) + list(values))

 
# Get the directory of the current script
dir_path = os.path.dirname(os.path.abspath(__file__))

# Create the full path to the file
file_path = os.path.join(dir_path, "term_output.csv")
write(file_path, data_dict)



plt.figure(figsize=(10, 10))
plt.hist([mpps_devs_i - cvx_devs_i for mpps_devs_i, cvx_devs_i in zip(mpps_devs, cvx_devs)], bins=np.arange(-0.16, 0.1, 0.02))
plt.xlabel(r'$\|\hat{X} - X_0\| - \|\bar{X} - X_0\|$')
plt.ylabel('Count')
# vertical line at 0 in red
plt.axvline(x=0, color='r', linestyle='--')
# tight layout
plt.tight_layout()
plt.savefig('figs/fig_4_paired_diffs_early.pdf')