# %%
import numpy as np
from snl import generateRandomData, solve_snl_fusion, getZfromNeighbors, getSNLProxData, solve_admm_double#, getBlockIncidence
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

def solver(n, a, d, dx, aa, Ni, Na, x, Z, primal, parallel=False, **kwargs):
    data, proxlist = getSNLProxData(n, a, d, dx, aa, Ni, Na)
    if not parallel:
        X, _, _ = solve(2*n, data, proxlist, Z=Z, W=Z, warmstartprimal=primal, verbose=False, **kwargs)
        log = [proxi.log for proxi in proxlist]
    else:
        X, _ = distributed_block_sparse_solve(2*n, data, proxlist, Z=Z, warmstartprimal=primal, verbose=False, logging=True, **kwargs)
        log = loadLogs(n)
    return X, log

class tol_cb():
    def __init__(self, maxitr, n, d=2):
        self.entries = (n + d)**2
        self.maxitr = maxitr

    def __call__(self, itr, all_x, all_y, **kwargs):
        if itr == self.maxitr-1:
            xbar = np.mean(all_x, axis=0)
            xdiff = np.linalg.norm(all_x - xbar)
            self.consensus_error = xdiff/(len(all_x) * self.entries)

            g = all_x - all_y
            sumg = np.sum(g, axis=0)
            self.subdifferential_sum_norm = np.linalg.norm(sumg)/self.entries

    
# %%
# '$||\\hat{X} - X||_F^2/n$'
def compare_errors_single(loglist, lognames, itrs, error_ref=None, metric=np.median, ylabel='MSE', reflabel='Relaxation solution', bounds=True, boundslabel=' IQR'):
    
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
consensus_errors_mpps = []
consensus_errors_mpps_warm = []
subg_errors_mpps = []
subg_errors_mpps_warm = []
consensus_errors_admm = []
consensus_errors_admm_warm = []
subg_errors_admm = []
subg_errors_admm_warm = []
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
    cb = tol_cb(itrs, n*2, d)
    X_mpps, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Z, blankprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=parallel, callback=cb)
    logs['Matrix Parametrized'].append(cleanlog)
    mpps_mse = np.sum((X_mpps[d:, :d] - x)**2)/n
    single_error = np.array([np.sum([((np.array(cleanlog[i+n][j][-d:]) + np.array(cleanlog[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
    errors_mpps.append(single_error)
    consensus_errors_mpps.append(cb.consensus_error)
    subg_errors_mpps.append(cb.subdifferential_sum_norm)
    print(f'{mpps_mse:.3e}', end='   ')

    # mpps with warm start
    cb = tol_cb(itrs, n*2, d)
    Xw_mpps, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Z, warmstartprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=parallel, callback=cb)
    logs['Warm Matrix Parametrized'].append(cleanlog)
    warm_mpps_mse = np.sum((Xw_mpps[d:, :d] - x)**2)/n
    single_error = np.array([np.sum([((np.array(cleanlog[i+n][j][-d:]) + np.array(cleanlog[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
    errors_mpps_warm.append(single_error)
    consensus_errors_mpps_warm.append(cb.consensus_error)
    subg_errors_mpps_warm.append(cb.subdifferential_sum_norm)
    print(f'{warm_mpps_mse:.3e}', end='   ')

    # ADMM
    cb = tol_cb(itrs, n*2, d)
    mean_X_admm, admm_log, all_X_admm = solve_admm_double(a, n, dx, aa, Ni, Na, warmstartprimal=blankprimal, alpha=admm_alpha, itrs=itrs, callback=cb)
    logs['ADMM'].append(admm_log)
    admm_mse = np.sum((mean_X_admm[d:, :d] - x)**2)/n
    single_error = np.array([np.sum([((np.array(admm_log[i+n][j][-d:]) + np.array(admm_log[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
    errors_admm.append(single_error)
    consensus_errors_admm.append(cb.consensus_error)
    subg_errors_admm.append(cb.subdifferential_sum_norm)
    print(f'{admm_mse:.3e}', end='   ')

    # ADMM with warm start
    cb = tol_cb(itrs, n*2, d)
    mean_X_admm, admm_log, all_X_admm = solve_admm_double(a, n, dx, aa, Ni, Na, warmstartprimal=admm_ws, alpha=admm_alpha, itrs=itrs, callback=cb)
    logs['Warm ADMM'].append(admm_log)
    warm_admm_mse = np.sum((mean_X_admm[d:, :d] - x)**2)/n
    single_error = np.array([np.sum([((np.array(admm_log[i+n][j][-d:]) + np.array(admm_log[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
    errors_admm_warm.append(single_error)
    consensus_errors_admm_warm.append(cb.consensus_error)
    subg_errors_admm_warm.append(cb.subdifferential_sum_norm)
    print(f'{warm_admm_mse:.3e}')
    rows.append([mpps_mse, warm_mpps_mse, admm_mse, warm_admm_mse])

print(f'Average consensus errors: MPPS - {np.mean(consensus_errors_mpps):.3e} ADMM - {np.mean(consensus_errors_admm):.3e} MPPS Warm - {np.mean(consensus_errors_mpps_warm):.3e} ADMM Warm - {np.mean(consensus_errors_admm_warm):.3e} ') 
print(f'Average norm of the sum of the subgradients: MPPS - {np.mean(subg_errors_mpps):.3e} ADMM - {np.mean(subg_errors_admm):.3e} MPPS Warm - {np.mean(subg_errors_mpps_warm):.3e} ADMM Warm - {np.mean(subg_errors_admm_warm):.3e} ')   

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