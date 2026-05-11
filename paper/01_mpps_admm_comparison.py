# %%
import numpy as np
import cvxpy as cp
from sys import path
# Sets parent directory as a string path
path.append('/home/peter.barkley/code/snl-nrl/code')
from snl import generateRandomData, solve_snl_fusion, getZfromNeighbors, getSNLProxData, solve_admm_double, getBlockIncidence
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from datetime import datetime
from collections import defaultdict
plt.rcParams['text.usetex'] = True
import os
if not os.path.exists('figs'): os.makedirs('figs')

# Change this to true if you want to use the MPI distributed version of the mpps algorithm
parallel = False
if parallel:
    # requires mpi and mpi4py
    from distributed_snl import distributed_block_sparse_solve
    from snl import loadLogs
else:
    # requires the development branch of oars
    from oars import solve

# %%
n = 30 # number of sensors
m = 6  # number of anchors
d = 2  # dimension of space
rd = 0.7 # neighborhood radius (sensors are in [0,1]^d), distances are only given within this radius
nf = 0.05 # noise factor. d_{ij} = d^0_{ij}(1 + nf * r_{ij}) where r_{ij} is a standard Gaussian rv
mpps_alpha = 10.0 # scaling parameter
mpps_gamma = 0.999 # step size parameter
admm_alpha = 150.0 # ADMM scaling parameter

itrs = 500
numtests = 50
logs = defaultdict(list)
cvx_mses = []

# %%
def getWarmstart(x, sigmas=[0.01, 0.01]):
    '''
    Generate a warm start set of x values perturbed from the given x

    Args:
        x (ndarray): n x d numpy array of the original x values
        sigmas (list): list of standard deviations for the perturbations

    Returns:
        ndarray: n x d numpy array of the perturbed x values

    '''

    n, d = x.shape
    return x + np.random.randn(n,d) * sigmas

def solver(n, a, d, dx, aa, Ni, Na, x, Z, primal, alpha=1.0, gamma=0.5, itrs=1000, parallel=False):
    data, proxlist = getSNLProxData(n, a, d, dx, aa, Ni, Na)
    if not parallel:
        X, log, _, _ = solve(2*n, data, proxlist, Z=Z, W=Z, warmstartprimal=primal, alpha=alpha, gamma=gamma, itrs=itrs, verbose=False)
    else:
        X, _ = distributed_block_sparse_solve(2*n, data, proxlist, Z=Z, warmstartprimal=primal, alpha=alpha, gamma=gamma, itrs=itrs, verbose=False, logging=True)
        log = loadLogs(n)
    return X, log


# %%
# '$||\\hat{X} - X||_F^2/n$'
def compare_errors_single(loglist, lognames, itrs, error_ref=None, metric=np.median, ylabel='Median MSE', reflabel='Relaxation solution', bounds=True, boundslabel=' IQR'):
    
    for logs, name in zip(loglist, lognames):
        line, = plt.plot([metric([log_i[j] for log_i in logs]) for j in range(itrs)], label=name)
        if bounds:
            lower = [np.percentile([log_i[j] for log_i in logs], 25) for j in range(itrs)]
            upper = [np.percentile([log_i[j] for log_i in logs], 75) for j in range(itrs)]
            color = line.get_color()
            plt.fill_between(range(itrs), lower, upper, alpha=0.5, color=color, label=name+boundslabel)

    # line at average cvx deviation
    if error_ref is not None:
        plt.axhline(y=error_ref, color='r', linestyle='--', label=reflabel)
    plt.legend()
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.xlabel('Iteration')

    return plt

# %%
print('date', 'time', 's', 'cvx_dev', 'mpps_mse', 'warm_mpps_mse', 'admm_mse', 'warm_admm_mse')
errors_mpps = []
errors_admm = []
errors_mpps_warm = []
errors_admm_warm = []
logs = defaultdict(list)
rows = []

for s in range(numtests):
    print(datetime.now(), s, end=' ')
    a, x, da, dx, aa, Ni, Na =generateRandomData(n, m, d, rd, nf, seed=s)

    
    # IP solution
    X_cvx, cvx_val = solve_snl_fusion(a, n, aa, dx, Ni, Na) # native MOSEK Fusion API
    # X_cvx, cvx_val = solve_snl_vec(a, n, aa, dx, Ni, Na) # CVXPY (slower, especially as size increases)
    cvx_mse = np.sum((X_cvx[d:, :d] - x)**2)/n
    cvx_mses.append(cvx_mse)
    print(f'{cvx_mse:.3e}', end='   ')

    # mpps setup
    sigmas = [0.2, 0.2]
    perturbedx = getWarmstart(x, sigmas=sigmas)
    warmstartprimal = np.block([[np.eye(d), perturbedx.T], [perturbedx, perturbedx @ perturbedx.T]])
    admm_ws = warmstartprimal.copy()
    blankprimal = np.zeros((n+d, n+d))
    blankprimal[:d, :d] = np.eye(d)
    Z = getZfromNeighbors(Ni)

    # mpps without warm start
    X_mpps, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Z, blankprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=parallel)
    logs['Matrix Parametrized'].append(cleanlog)
    mpps_mse = np.sum((X_mpps[d:, :d] - x)**2)/n
    single_error = np.array([np.sum([((np.array(cleanlog[i+n][j][-d:]) + np.array(cleanlog[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
    errors_mpps.append(single_error)
    print(f'{mpps_mse:.3e}', end='   ')

    # mpps with warm start
    Xw_mpps, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Z, warmstartprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=parallel)
    logs['Warm Matrix Parametrized'].append(cleanlog)
    warm_mpps_mse = np.sum((Xw_mpps[d:, :d] - x)**2)/n
    single_error = np.array([np.sum([((np.array(cleanlog[i+n][j][-d:]) + np.array(cleanlog[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
    errors_mpps_warm.append(single_error)
    print(f'{warm_mpps_mse:.3e}', end='   ')

    # ADMM
    mean_X_admm, admm_log, all_X_admm = solve_admm_double(a, n, dx, aa, Ni, Na, warmstartprimal=blankprimal, alpha=admm_alpha, itrs=itrs)
    logs['ADMM'].append(admm_log)
    admm_mse = np.sum((mean_X_admm[d:, :d] - x)**2)/n
    single_error = np.array([np.sum([((np.array(admm_log[i+n][j][-d:]) + np.array(admm_log[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
    errors_admm.append(single_error)
    print(f'{admm_mse:.3e}', end='   ')

    # ADMM with warm start
    mean_X_admm, admm_log, all_X_admm = solve_admm_double(a, n, dx, aa, Ni, Na, warmstartprimal=admm_ws, alpha=admm_alpha, itrs=itrs)
    logs['Warm ADMM'].append(admm_log)
    warm_admm_mse = np.sum((mean_X_admm[d:, :d] - x)**2)/n
    single_error = np.array([np.sum([((np.array(admm_log[i+n][j][-d:]) + np.array(admm_log[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
    errors_admm_warm.append(single_error)
    print(f'{warm_admm_mse:.3e}')
    rows.append([mpps_mse, warm_mpps_mse, admm_mse, warm_admm_mse])

    

ref = np.median(cvx_mses)

# error cold
fig = compare_errors_single([errors_mpps, errors_admm], ['Matrix Parametrized', 'ADMM'], itrs, ref)
fig.savefig('figs/fig_1_mpps_admm_comparison.pdf', bbox_inches='tight')
fig.close()

# error warm
fig = compare_errors_single([errors_mpps_warm, errors_admm_warm], ['Matrix Parametrized - Warmstart', 'ADMM - Warmstart'], itrs, ref)
fig.savefig('figs/fig_1_mpps_admm_comparison_warm.pdf', bbox_inches='tight')
fig.close()

import pandas as pd
df = pd.DataFrame(rows, columns=['mpps_mse', 'warm_mpps_mse', 'admm_mse', 'warm_admm_mse'])
df.to_csv('mpps_admm_comparison.csv', index=False)