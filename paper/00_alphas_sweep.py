# %%
# %%
import numpy as np
from sys import path
from pathlib import Path
# Sets parent directory as a string path
parent_dir = str(Path(__file__).resolve().parent.parent)
path.append(parent_dir)
from snl import generateRandomData, solve_snl_fusion, getZfromNeighbors, getSNLProxData, loadLogs, solve_admm_double
from oars import solve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from collections import defaultdict
from datetime import datetime
import pandas as pd



# %%
n = 30
m = 6
d = 2

# %%
numtests = 50

# %%

oarslogs = defaultdict(list)
oars_devs = defaultdict(list)
cvx_vals = []
cvx_devs = []
cvx_errors = []
x_vals = []
cvx_X_vals = []
itrs = {}
itrs['MPPS'] = 800
itrs['ADMM'] = 2400
alphas = {}
alphas['MPPS'] = [1, 5, 10, 15, 20, 25]
alphas['ADMM'] = [1., 10., 100., 125., 140., 145., 150., 155., 160., 175., 200., 225.]
gamma = .999
min_errors = defaultdict(list)
min_errors_idx = defaultdict(list)
sum_errors = defaultdict(list)
errors = defaultdict(list)
rows = []
# %%
# print(datetime.now(), 's', 'cvx', alphas)
for s in range(numtests):
    a, x, da, dx, aa, Ni, Na = generateRandomData(n, m, d, 0.7, .05, seed=s)
    x_vals.append(x)

    # CVX
    X_cvx, cvx_val = solve_snl_fusion(a, n, aa, dx, Ni, Na)
    cvx_dev = np.linalg.norm(X_cvx[d:, :d] - x, 'fro')
    # cvx_error = cvx_dev
    print(datetime.now(), s, np.round(cvx_dev,3))
    cvx_vals.append(cvx_val)
    cvx_devs.append(cvx_dev)
    cvx_X_vals.append(X_cvx[d:, :d])

    # OARS setup
    Z = getZfromNeighbors(Ni)

    # OARS without warm start
    for alg in ['MPPS', 'ADMM']:
        print(alg)
        for alpha in alphas[alg]:

            blankprimal = np.zeros((n+d, n+d))
            blankprimal[:d, :d] = np.eye(d)
            data, proxlist = getSNLProxData(n, a, d, dx, aa, Ni, Na) #n, a, d, dx, aa, Ni, Na
            if alg == 'MPPS':
                alg_x, alg_log, _, _ = solve(2*n, data, proxlist, Z=Z, W=Z, warmstartprimal=blankprimal, alpha=alpha, gamma=gamma, itrs=itrs[alg], verbose=False)
            else: 
                alg_x, alg_log, _ = solve_admm_double(a, n, dx, aa, Ni, Na, warmstartprimal=blankprimal, alpha=alpha, itrs=itrs[alg])
        
            single_error = np.array([np.sum([((np.array(alg_log[i+n][j][-d:]) + np.array(alg_log[i][j][-d:]) )/2 - x[i])**2 for i in range(n)])**0.5 for j in range(itrs[alg])])
            error_integral = np.sum(single_error)
            min_error = min(single_error)
            min_err_idx = np.argmin(single_error)
            min_errors[(alg, alpha)].append(min_error)
            min_errors_idx[(alg, alpha)].append(min_err_idx)
            sum_errors[(alg, alpha)].append(error_integral)
            errors[(alg, alpha)].append(single_error)
            print(f'{alpha} {min_error:.3f} {error_integral:.3f} {min_err_idx}')
            rows.append({
                'Algorithm': alg,
                'Alpha': alpha,
                'Trial': s,
                'Min_Error': min_error,
                'Min_Error_Idx': min_err_idx,
                'Sum_Error': error_integral,
            })



df = pd.DataFrame(rows)
df.to_csv('errors_data.csv')
