# %%
import numpy as np
import cvxpy as cp
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
    from distributed_block import distributed_block_sparse_solve
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
cvx_vals = []
cvx_mean_devs = []
cvx_devs = []
cvx_errors = []
x_vals = []

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

def relative_error(logs, x_vals, j, n, d=2, psd=False):
    if psd==False:
        offset = n
    else:
        offset = 0
    return [sum(np.linalg.norm(np.array(log[i+offset][j][-d:]) - x_vals[lidx][i])**2 for i in range(n))**0.5/np.linalg.norm(x_vals[lidx]) for lidx, log in enumerate(logs)]

def deviation(logs, x_vals, j, n, d=2, psd=False):
    if psd==False:
        offset = n
    else:
        offset = 0
    return [sum(np.linalg.norm(np.array(log[i+offset][j][-d:]) - x_vals[lidx][i])**2 for i in range(n))**0.5 for lidx, log in enumerate(logs)]

def mean_deviation(logs, x_vals, j, n, d=2, psd=False):
    if psd==False:
        offset = n
    else:
        offset = 0
    return [np.mean([np.linalg.norm(np.array(log[i+offset][j][-d:]) - x_vals[lidx][i]) for i in range(n)]) for lidx, log in enumerate(logs)]

def getCentrality(x, center):
    return np.mean([np.linalg.norm(pt - center) for pt in x])

def compare_errors(loglist, lognames, n, itrs, x_vals, error_ref=None, metric=np.median, error=relative_error, ylabel='$\\frac{||\\hat{X} - X_0||_F}{||X_0||_F}$', reflabel='Relaxation solution', bounds=None, boundslabel=' IQR'):
    
    for logs, name in zip(loglist, lognames):
        line, = plt.plot([metric(error(logs, x_vals, j, n)) for j in range(itrs)], label=name)
        if bounds is not None:
            lower = [bounds[0](error(logs, x_vals, j, n)) for j in range(itrs)]
            upper = [bounds[1](error(logs, x_vals, j, n)) for j in range(itrs)]
            color = line.get_color()
            plt.fill_between(range(itrs), lower, upper, alpha=0.5, color=color, label=name+boundslabel)

    # line at average cvx deviation
    if error_ref is not None:
        plt.axhline(y=error_ref, color='r', linestyle='--', label=reflabel)
    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel('Iteration')
    # plt.title('SNL mean total deviation from true value relative to MOSEK solution')

    return plt

def solver(n, a, d, dx, aa, Ni, Na, x, Z, primal, alpha=1.0, gamma=0.5, itrs=1000, parallel=False):
    data, proxlist = getSNLProxData(n, a, d, dx, aa, Ni, Na)
    if not parallel:
        X, log, _, _ = solve(2*n, data, proxlist, Z=Z, W=Z, warmstartprimal=primal, alpha=alpha, gamma=gamma, itrs=itrs, verbose=False)
    else:
        X, _ = distributed_block_sparse_solve(2*n, data, proxlist, Z=Z, warmstartprimal=primal, alpha=alpha, gamma=gamma, itrs=itrs, verbose=False, logging=True)
        log = loadLogs(n)
    return X, log

print('date', 'time', 's', 'cvx_dev', 'mpps_dev', 'admm_dev', 'warm_mpps_dev', 'warm_admm_dev')

for s in range(numtests):
    print(datetime.now(), s, end=' ')
    a, x, da, dx, aa, Ni, Na =generateRandomData(n, m, d, rd, nf, seed=s)
    x_vals.append(x)
    xnorm = np.linalg.norm(x)

    # IP solution
    X_cvx, cvx_val = solve_snl_fusion(a, n, aa, dx, Ni, Na) # native MOSEK Fusion API
    # X_cvx, cvx_val = solve_snl_vec(a, n, aa, dx, Ni, Na) # CVXPY (slower, especially as size increases)
    cvx_dev = np.linalg.norm(X_cvx[d:, :d] - x, 'fro')
    cvx_devs.append(cvx_dev)
    mean_dev = np.mean([np.linalg.norm(X_cvx[i+d, :d] - x[i]) for i in range(n)])
    cvx_mean_devs.append(mean_dev)
    cvx_error = cvx_dev/xnorm
    cvx_errors.append(cvx_error)
    print(f'{cvx_dev:.3f} {cvx_error:.3f} {mean_dev:.3f}', end='   ')

    # mpps setup
    sigmas = [0.2, 0.2]
    perturbedx = getWarmstart(x, sigmas=sigmas)
    warmstartprimal = np.block([[np.eye(d), perturbedx.T], [perturbedx, perturbedx @ perturbedx.T]])
    admm_ws = warmstartprimal.copy()
    blankprimal = np.zeros((n+d, n+d))
    blankprimal[:d, :d] = np.eye(d)
    Z = getZfromNeighbors(Ni)

    # mpps without warm start
    X_mpps, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Z, blankprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=False)
    mpps_dev = np.linalg.norm(X_mpps[d:, :d] - x, 'fro')
    mean_dev = np.mean([np.linalg.norm(X_mpps[i+d, :d] - x[i]) for i in range(n)])
    logs['Matrix Parametrized'].append(cleanlog)
    print(f'{mpps_dev:.3f} {mpps_dev/xnorm:.3f} {mean_dev:.3f}', end='   ')

    # mpps with warm start
    Xw_mpps, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Z, warmstartprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=False)
    mpps_dev = np.linalg.norm(Xw_mpps[d:, :d] - x, 'fro')
    mean_dev = np.mean([np.linalg.norm(Xw_mpps[i+d, :d] - x[i]) for i in range(n)])
    logs['Warm Matrix Parametrized'].append(cleanlog)
    print(f'{mpps_dev:.3f} {mpps_dev/xnorm:.3f} {mean_dev:.3f}', end='   ')

    # ADMM
    mean_X_admm, admm_log, all_X_admm = solve_admm_double(a, n, dx, aa, Ni, Na, warmstartprimal=blankprimal, alpha=admm_alpha, itrs=itrs)
    dev = np.linalg.norm(mean_X_admm[d:, :d] - x, 'fro')
    mean_dev = np.mean([np.linalg.norm(mean_X_admm[i+d, :d] - x[i]) for i in range(n)])
    logs['ADMM'].append(admm_log)
    print(f'{dev:.3f} {dev/xnorm:.3f} {mean_dev:.3f}', end='   ')

    # ADMM with warm start
    mean_X_admm, admm_log, all_X_admm = solve_admm_double(a, n, dx, aa, Ni, Na, warmstartprimal=admm_ws, alpha=admm_alpha, itrs=itrs)
    dev = np.linalg.norm(mean_X_admm[d:, :d] - x, 'fro')
    mean_dev = np.mean([np.linalg.norm(mean_X_admm[i+d, :d] - x[i]) for i in range(n)])
    logs['Warm ADMM'].append(admm_log)
    print(f'{dev:.3f} {dev/xnorm:.3f} {mean_dev:.3f}')

# Relative error cold
fig = compare_errors([logs['Matrix Parametrized'], logs['ADMM']], ['Matrix Parametrized', 'ADMM'], n, itrs, x_vals, np.median(cvx_errors), bounds=(lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)))

# save plot as pdf
fig.savefig('figs/fig_1_mpps_admm_comparison_errors_median_bounds.pdf', bbox_inches='tight')
fig.close()

# Relative error warm
fig = compare_errors([logs['Warm Matrix Parametrized'], logs['Warm ADMM']], ['Warm Matrix Parametrized', 'Warm ADMM'], n, itrs, x_vals, np.median(cvx_errors), bounds=(lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)))

# save plot as pdf
fig.savefig('figs/fig_1_mpps_admm_comparison_errors_median_bounds_warm.pdf', bbox_inches='tight')
fig.close()

# Deviation cold
fig = compare_errors([logs['Matrix Parametrized'], logs['ADMM']], ['Matrix Parametrized', 'ADMM'], n, itrs, x_vals, np.median(cvx_devs), metric=np.median, error=deviation, ylabel='$||\\hat{X} - X_0||$', bounds=(lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)))

# save plot as pdf
fig.savefig('figs/fig_1_mpps_admm_comparison_dev_median_bounds.pdf', bbox_inches='tight')
fig.close()

# Deviation warm
fig = compare_errors([logs['Warm Matrix Parametrized'], logs['Warm ADMM']], ['Warm Matrix Parametrized', 'Warm ADMM'], n, itrs, x_vals, np.median(cvx_devs), metric=np.median, error=deviation, ylabel='$||\\hat{X} - X||_F$', bounds=(lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)))

# save plot as pdf
fig.savefig('figs/fig_1_mpps_admm_comparison_dev_median_bounds_warm.pdf', bbox_inches='tight')
fig.close()

# Mean deviation cold
fig = compare_errors([logs['Matrix Parametrized'], logs['ADMM']], ['Matrix Parametrized', 'ADMM'], n, itrs, x_vals, np.median(cvx_mean_devs), metric=np.median, error=mean_deviation, ylabel='$\\frac{1}{n}\\sum_{i=1}^n||\\hat{X}_i - X_i||$', bounds=(lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)))

# save plot as pdf
fig.savefig('figs/fig_1_mpps_admm_comparison_meandev_median_bounds.pdf', bbox_inches='tight')
fig.close()

# Deviation warm
fig = compare_errors([logs['Warm Matrix Parametrized'], logs['Warm ADMM']], ['Warm Matrix Parametrized', 'Warm ADMM'], n, itrs, x_vals, np.median(cvx_mean_devs), metric=np.median, error=mean_deviation, ylabel='$\\frac{1}{n}\\sum_{i=1}^n||\\hat{X}_i - X_i||$', bounds=(lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)))

# save plot as pdf
fig.savefig('figs/fig_1_mpps_admm_comparison_meandev_median_bounds_warm.pdf', bbox_inches='tight')
fig.close()

#####
print("Matrix Design Test and Centrality Test")

numtests = 20
logs = defaultdict(list)
x_vals = []
centers = [] 
ip_centralities = [] 
actual_centralities = [] 

def getMinResist_Fast(n, M, c=None, **kwargs):
    '''
    Get two block Z matrix over the links in Ni with minimal SLEM on the matrix I-Z/2

    Args:
        n (int): number of nodes
        Ni (list): list of lists of neighbors for each node
        c (float): optional connectivity parameter (default 2*(1-cos(pi/n)))
        kwargs: additional keyword arguments for the solver
        
    Returns:
        Z (ndarray): 2n x 2n matrix
        v (float): optimal value

    '''
    if c is None:
        c = 2*(1-np.cos(np.pi/n))

    ez = M.shape[0]

    # Variables
    z = cp.Variable(ez, nonneg=True)
    Z = M.T @ cp.diag(z) @ M

    # Constraints
    ones = np.ones((2*n, 2*n))
    cons = [Z + 4*ones - c*np.eye(2*n) >> 0, # Z is connected
            cp.diag(Z) == 2] # Z diagonal is 2

    # Solve
    terz = cp.tr_inv(Z + (1/(2*n))*ones)
    obj = cp.Minimize(terz)
    
    # Solve
    prob = cp.Problem(obj, cons)
    prob.solve(**kwargs)

    return Z.value, prob.value

def getMinSLEM_Fast(n, M, c=None, **kwargs):
    '''
    Get two block Z matrix over the links in Ni with minimal SLEM on the matrix I-Z/2

    Args:
        n (int): number of nodes
        Ni (list): list of lists of neighbors for each node
        c (float): optional connectivity parameter (default 2*(1-cos(pi/n)))
        kwargs: additional keyword arguments for the solver
        
    Returns:
        Z (ndarray): 2n x 2n matrix
        v (float): optimal value

    '''
    if c is None:
        c = 2*(1-np.cos(np.pi/n))

    ez = M.shape[0]

    # Variables
    z = cp.Variable(ez, nonneg=True)
    Z = M.T @ cp.diag(z) @ M
    gz = cp.Variable(1)

    # Constraints
    ones = np.ones((2*n, 2*n))
    ZP = np.eye(2*n) - Z/2 # Find adjacency matrix of Z scaled to sum to 1, and with diagonal entries <= 1
    ZPvU = ZP - (1/(2*n))*ones # difference between scaled Z graph and uniform graph
    cons = [Z + 4*ones - c*np.eye(2*n) >> 0, # Z is connected
            cp.diag(Z) == 2, # Z diagonal is 2
            -gz*np.eye(2*n) << ZPvU, ZPvU << gz*np.eye(2*n)] # ZPvU is bounded by gz

    # Solve
    obj = cp.Minimize(gz)
    
    # Solve
    prob = cp.Problem(obj, cons)
    prob.solve(**kwargs)

    return Z.value, prob.value    

def getMaxFiedler_Fast(n, M, feasibility=False, reduce=False, c=None, gamma=1.0, **kwargs):
    '''
    Get core variables and constraints for the algorithm design SDP

    :math:`W \\mathbb{1} = 0`

    :math:`Z \\mathbb{1} = 0`

    :math:`\\lambda_{1}(W) + \\lambda_{2}(W) \\geq c`

    :math:`Z - W \\succeq 0`

    :math:`\\mathrm{diag}(Z) = Z_{11}\\mathbb{1}`

    :math:`2 - \\varepsilon \\leq Z_{11} \\leq 2 + \\varepsilon`

    Args:
        n (int): number of nodes
        Ni (list): list of lists of neighbors for each node
        maxFiedler (bool): whether to maximize the Fiedler value
        c (float): connectivity parameter (default 2*(1-cos(pi/n)))
        eps (float): nonnegative epsilon for Z[0,0] = 2 + eps constraint (default 0.0)
        gamma (float): scaling parameter for Z (default 1.0)
        kwargs: additional keyword arguments for the solver
        
    Returns:
        Z (ndarray): 2n x 2n matrix
        v (float): optimal value

    '''
    ez = M.shape[0]

    # Variables
    z = cp.Variable(ez, nonneg=True)
    Z = M.T @ cp.diag(z) @ M

    if not feasibility:
        c = cp.Variable()
        obj_exp = c
        if reduce:
            obj_exp = c - cp.norm(z+2, 2)
        obj = cp.Maximize(obj_exp)
    else:
        if c is None:
            c = 2*(1-np.cos(np.pi/n))
        obj = cp.Minimize(0)


    # Constraints
    ones = 4*np.outer(np.ones(2*n), np.ones(2*n))
    cons = [Z + ones - c*np.eye(2*n) >> 0,
            cp.diag(Z) == 2]

    # Solve
    prob = cp.Problem(obj, cons)
    prob.solve(**kwargs)

    return Z.value, prob.value    

def plot_error(logdict, designs, x_vals, n, d=2, metric=np.mean, logscale=False):
    fig, ax = plt.subplots()
    markers = ['X', 's', 'D', '^', 'v', 'p', 'o', '*', 'P', 'd']
    lidx = 0
    for key in designs:
        log = logdict[key]
        ax.plot([metric(relative_error(log, x_vals, j, n, d)) for j in range(1, itrs)], label=key, marker=markers[lidx], alpha=0.7, markevery=(50+lidx*15, 200))
        lidx += 1

    ax.legend()
    if logscale: ax.set_yscale('log')
    
    ax.set_ylim(0.05, 0.1)
    ax.grid(True, which="both", ls="-")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$\\frac{||\\hat{X} - X_0||_F}{||X_0||_F}$')
    fig.tight_layout()
    return fig


print('date', 'time', 'test', 'mosek',  'sk', 'con', 'resist', 'slem')
for s in range(numtests):
    print(datetime.now(), s, end=' ')
    a, x, da, dx, aa, Ni, Na = generateRandomData(n, m, d, rd, nf, seed=s)
    
    Mz = getBlockIncidence(Ni)
    xnorm = np.linalg.norm(x)
    x_vals.append(x)
    center = np.mean(a, axis=0)
    centers.append(center)
    actual_centralities.append(getCentrality(x, center))

    # Mosek
    X_cvx, cvx_val = solve_snl_fusion(a, n, aa, dx, Ni, Na)
    cvx_dev = np.linalg.norm(X_cvx[d:, :d] - x, 'fro')/xnorm
    print(np.round(cvx_dev,3), end=' ')
    ip_centralities.append(getCentrality(X_cvx[d:, :d], center))

    # SK
    blankprimal = np.zeros((n+d, n+d))
    blankprimal[:d, :d] = np.eye(d)
    Z = getZfromNeighbors(Ni)
    X_mpps, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Z, blankprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=False)
    mpps_err = np.linalg.norm(X_mpps[d:, :d] - x, 'fro')/xnorm
    # cleanlog = loadLogs(n)
    print(f'{mpps_err:.3f}', end=' ')
    logs['Sinkhorn-Knopp'].append(cleanlog)

    # Center Warmstart
    primal = blankprimal.copy()
    centeredX = np.tile(center, (n, 1))
    primal[:d, d:] = centeredX.T
    primal[d:, :d] = centeredX
    primal[d:, d:] = centeredX @ centeredX.T
    X, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Z, primal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=False)
    logs['Centered'].append(cleanlog)

    # Max Connectivity
    Zc, _ = getMaxFiedler_Fast(n, Mz)
    X, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Zc, blankprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=False)
    mpps_dev = np.linalg.norm(X[d:, :d] - x, 'fro')/xnorm
    print(np.round(mpps_dev,3), end=' ')
    logs['Max Connectivity'].append(cleanlog)

    # Min Resist
    blankprimal = np.zeros((n+d, n+d))
    blankprimal[:d, :d] = np.eye(d)
    Zr, _ = getMinResist_Fast(n, Mz)
    X, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Zr, blankprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=False)
    mpps_dev = np.linalg.norm(X[d:, :d] - x, 'fro')/xnorm
    print(np.round(mpps_dev,3), end=' ')
    logs['Min Resistance'].append(cleanlog)

    # Min SLEM    
    blankprimal = np.zeros((n+d, n+d))
    blankprimal[:d, :d] = np.eye(d)
    Zs, _ = getMinSLEM_Fast(n, Mz)
    X, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Zs, blankprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=False)
    mpps_dev = np.linalg.norm(X[d:, :d] - x, 'fro')/xnorm
    print(np.round(mpps_dev,3))
    logs['Min SLEM'].append(cleanlog)


designs = ['Sinkhorn-Knopp', 'Max Connectivity', 'Min Resistance', 'Min SLEM']
fig = plot_error(logs, designs, x_vals, n)

fig.savefig('figs/fig_2_sk_oars_error.pdf')
plt.close()

metric = np.mean
plt.plot([metric([np.mean([np.linalg.norm(np.array(entry[j][-d:]) - center) for entry in log]) for log, center in zip(logs['Centered'], centers)]) for j in range(itrs)], label='Matrix Parametrized')

# Calculate CVX dev
cvx_centrality_metric = metric(ip_centralities)
actual_centrality_metric = metric(actual_centralities)

# Add line for CVX
plt.axhline(y=cvx_centrality_metric, color='r', linestyle='--', label='Relaxation Solution')

# Add line for actual
plt.axhline(y=actual_centrality_metric, color='g', linestyle='--', label='Actual')

plt.xlabel('Iteration')
plt.ylabel('Mean distance to anchor center of mass')
plt.legend()

# save figure
plt.savefig('figs/fig_3_centroid_distance.pdf')
plt.close()


####
# Quick Matrix
print("Matrix design time")

from pandas import DataFrame
from time import time
from oars.matrices import getBlockFixed, getMinResist, getMinSLEM, getMaxConnectivity

def getCore(n, M, maxFiedler=True, c=None, gamma=1.0, **kwargs):
    '''
    Get core variables and constraints for the algorithm design SDP

    :math:`W \\mathbb{1} = 0`

    :math:`Z \\mathbb{1} = 0`

    :math:`\\lambda_{1}(W) + \\lambda_{2}(W) \\geq c`

    :math:`Z - W \\succeq 0`

    :math:`\\mathrm{diag}(Z) = Z_{11}\\mathbb{1}`

    :math:`2 - \\varepsilon \\leq Z_{11} \\leq 2 + \\varepsilon`

    Args:
        n (int): number of nodes
        !Ni (list): list of lists of neighbors for each node
        M (array): d x n array for edge-node incidence
        maxFiedler (bool): whether to maximize the Fiedler value
        c (float): connectivity parameter (default 2*(1-cos(pi/n)))
        eps (float): nonnegative epsilon for Z[0,0] = 2 + eps constraint (default 0.0)
        gamma (float): scaling parameter for Z (default 1.0)
        kwargs: additional keyword arguments for the solver
        
    Returns:
        Z (ndarray): 2n x 2n matrix
        v (float): optimal value

    '''
    if maxFiedler:
        c = cp.Variable()
        obj = cp.Maximize(c)
    else:
        if c is None:
            c = 2*(1-np.cos(np.pi/n))
        obj = cp.Minimize(0)


    # Mz = snl.getBlockIncidence(Ni)
    ez = M.shape[0]

    # Variables
    z = cp.Variable(ez, nonneg=True)
    Z = M.T @ cp.diag(z) @ M

    # Constraints
    ones = 4*np.outer(np.ones(2*n), np.ones(2*n))
    cons = [Z + ones - c*np.eye(2*n) >> 0,
            cp.diag(Z) == 2]

    # Solve
    prob = cp.Problem(obj, cons)
    prob.solve(**kwargs)

    return Z.value, prob.value    

def fixedZ(n, Ni):
    Z_fixed, _ = getBlockFixed(2*n, n)
    for i in range(n):
        for j in range(n):
            if i != j and j not in Ni[i] and i not in Ni[j]:
                Z_fixed[i, n+j] = 0
    return Z_fixed

def getZfromGraphPI(M):
    n, d = M.shape
    Ma = np.abs(M)
    w = np.linalg.pinv(Ma)@np.ones(n)
    return 2*(M @ np.diag(w) @ M.T)

ns = [10, 15, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350] 
numtests = 10
results = []
connectivity_off = False
resistance_off = False
slem_off = False
cutoff_time = 12

for n in ns:
    for s in range(numtests):
        print(n, s, end=' ')
        a, x, da, dx, aa, Ni, Na = generateRandomData(n, 5, 2, 0.7, .02, cutoff=min(5, n//4), seed=s)
        M = getBlockIncidence(Ni)

        # SK
        start = time()
        Z = getZfromNeighbors(Ni)
        end = time()
        ti = end - start
        print(f'{ti:.3f}', end=' ')

        # Feasibility
        start = time()
        Z, v = getCore(n, M, maxFiedler=False)
        end = time()
        tf = end - start
        print(f'{tf:.3f}', end=' ')

        # Improved Max Fiedler 
        start = time()
        Z, v = getCore(n, M)
        end = time()
        tm = end - start
        print(f'{tm:.3f}', end=' ')

        # Max Connectivity
        Z_fixed = fixedZ(n, Ni)
        if not connectivity_off:
            start = time()
            Z, W = getMaxConnectivity(2*n, fixed_Z=Z_fixed, fixed_W=Z_fixed, verbose=False)
            end = time()
            tc = end - start
            connectivity_off = (tc > cutoff_time)
            print(f'{tc:.3f}', end=' ')
        else:
            tc = None
            print('X    ', end=' ')

        # Min SLEM
        if not slem_off:
            start = time()
            Z, W = getMinSLEM(2*n, fixed_Z=Z_fixed, fixed_W=Z_fixed, verbose=False)
            end = time()
            ts = end - start
            slem_off = (ts > cutoff_time)
            print(f'{ts:.3f}', end=' ')
        else:
            ts = None
            print('X    ', end=' ')

        # Min Resistance
        if not resistance_off:
            start = time()
            Z, W = getMinResist(2*n, fixed_Z=Z_fixed, fixed_W=Z_fixed, verbose=False)
            end = time()
            tr = end - start
            resistance_off = (tr > cutoff_time)
            print(f'{tr:.3f}', end=' ')
        else:
            tr = None
            print('X    ', end=' ')

        results.append((n, tr, ts, tc, tm, tf, ti))
        print()

algnames = ['OARS Min Resistance', 'OARS Min SLEM', 'OARS Max Connectivity', 'Improved Max Connectivity', 'Feasibility', 'Sinkhorn-Knopp']
df = DataFrame(results, columns=['Nodes'] + algnames)

# Plot the mean time for each method as a function of n
plt.plot(df.groupby('Nodes').mean())
plt.ylim(-0.5, 10.5)
plt.xlabel('Nodes')
plt.ylabel('Time (s)')
plt.legend(algnames)
plt.savefig('figs/fig_2_design_time.pdf')
plt.close()


# %%
########
# Point Plot
print("Point Plot")
from snl import node_objective_value
s = 4
n = 30
itrs = 500

def plotPoints(a, x, xhat=None, xbar=None, link=False, title=None, legend=True):
    
    fig, axs = plt.subplots(ncols=1)
    axs.scatter(a[:,0], a[:,1], c='r', label='Anchor points')
    axs.scatter(x[:,0], x[:,1], c='b', label='Sensor points')



    axs.scatter(xhat[:,0], xhat[:,1], c='g', label='Early Termination')
    axs.scatter(xbar[:,0], xbar[:,1], c='k', label='Relaxation Solution')

    for i in range(len(x)):
        # Dashed line for link
        axs.plot([x[i,0], xbar[i,0]], [x[i,1], xbar[i,1]], 'k', linestyle='dashed')

    fartherlabelset = False
    closerlabelset = False
    for i in range(len(x)):
        if np.linalg.norm(x[i] - xhat[i]) > np.linalg.norm(x[i] - xbar[i]):
            color = 'r'
            label = 'Farther'
        else:
            color = 'g'
            label = 'Closer'
        if not fartherlabelset and label == 'Farther':
            fartherlabelset = True
            axs.plot([x[i,0], xhat[i,0]], [x[i,1], xhat[i,1]], color, label=label)
        elif not closerlabelset and label == 'Closer':
            closerlabelset = True
            axs.plot([x[i,0], xhat[i,0]], [x[i,1], xhat[i,1]], color, label=label)
        else:
            axs.plot([x[i,0], xhat[i,0]], [x[i,1], xhat[i,1]], color)
    
    # Add legend outside the plot
    fig.subplots_adjust(bottom=0.3, wspace=0.33)

    # add centered legend below the plot with the points and line entries for green and red
    if legend:
       fig.legend(loc='upper center', 
                bbox_to_anchor=(0.5, 0.0),fancybox=False, shadow=False, ncol=3)
       
    fig.tight_layout()
    return fig

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


    def __call__(self, itr, all_x, all_v):
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

a, x, da, dx, aa, Ni, Na = generateRandomData(n, m, d, rd, nf, seed=s)
center = np.mean(a, axis=0)

# CVX
X_cvx, cvx_val = solve_snl_fusion(a, n, aa, dx, Ni, Na)
cvx_X_val = X_cvx[d:, :d]

# mpps setup
primal = np.zeros((n+d, n+d))
primal[:d, :d] = np.eye(d)
centeredX = np.tile(center, (n, 1))
primal[:d, d:] = centeredX.T
primal[d:, :d] = centeredX
primal[d:, d:] = centeredX @ centeredX.T
Z = getZfromNeighbors(Ni)

cb = callback(n, d, dx, a, aa, Ni, Na)
data, proxlist = getSNLProxData(n, a, d, dx, aa, Ni, Na)
X_mpps, cleanlog, _, _ = solve(2*n, data, proxlist, W=Z, Z=Z, warmstartprimal=primal, alpha=mpps_alpha, gamma=1.0, itrs=itrs, verbose=False, callback=cb)
print(len(cleanlog[0]))

xhat = np.zeros_like(x)
for i in range(n):
    xhat[i] = cleanlog[i+n][-1][-d:]
fig = plotPoints(a, x, xhat=xhat, xbar=cvx_X_val, link=True)
fig.savefig('figs/fig_3_point_plot_dist.pdf', bbox_inches='tight')
plt.close()

fig = plotPoints(a, x, xhat=X_mpps[d:, :d], xbar=cvx_X_val, link=True)
fig.savefig('figs/fig_3_point_plot.pdf', bbox_inches='tight')
plt.close()


# %%
########
# Termination Comparison
import csv
import os

print("Termination Comparison")
itrs = 800
numtests = 300
cvx_vals = []
cvx_devs = []
cvx_mean_devs = []
cvx_errors = []
mpps_devs = []
mpps_mean_devs = []
mpps_devs_dist = []
mpps_mean_devs_dist = []
x_vals = []
results_list = []

X_mpps_dist = np.zeros((n,d))
print('date\t', 'time', 's', 'cvx_dev', 'cvx_rel_error', 'mpps_dev', 'mpps_rel_error', 'itrs')
for s in range(numtests):
    print(datetime.now(), s, end='\t')
    a, x, da, dx, aa, Ni, Na = generateRandomData(n, m, d, .7, .05, seed=s)
    x_vals.append(x)
    center = np.mean(a, axis=0)
    xnorm = np.linalg.norm(x)

    # CVX
    X_cvx, cvx_val = solve_snl_fusion(a, n, aa, dx, Ni, Na)
    cvx_dev = np.linalg.norm(X_cvx[d:, :d] - x, 'fro')
    cvx_vals.append(cvx_val)
    cvx_rel_dev = cvx_dev/xnorm
    cvx_devs.append(cvx_dev)
    cvx_mean_dev = np.mean([np.linalg.norm(X_cvx[i+d, :d] - x[i]) for i in range(n)])
    cvx_mean_devs.append(cvx_mean_dev)
    cvx_errors.append(cvx_rel_dev)
    print(f'{cvx_dev:.3f}', end='\t')

    # mpps
    Z = getZfromNeighbors(Ni)
    data, proxlist = getSNLProxData(n, a, d, dx, aa, Ni, Na)
    primal = np.zeros((n+d, n+d))
    primal[:d, :d] = np.eye(d)
    cb = callback(n, d, dx, a, aa, Ni, Na)
    X_mpps, cleanlog, _, _ = solve(2*n, data, proxlist, W=Z, Z=Z, warmstartprimal=primal, alpha=mpps_alpha, gamma=1.0, itrs=itrs, verbose=False, callback=cb)
    mpps_dev = np.linalg.norm(X_mpps[d:, :d] - x, 'fro')
    mpps_devs.append(mpps_dev)
    mpps_mean_dev = np.mean([np.linalg.norm(X_mpps[i+d, :d] - x[i]) for i in range(n)])
    mpps_mean_devs.append(mpps_mean_dev)
    itr_stop = len(cleanlog[0])
    for i in range(n):
        X_mpps_dist[i] = cleanlog[i+n][itr_stop-1][-d:]
    mpps_dev_dist = np.linalg.norm(X_mpps_dist - x, 'fro')
    mpps_devs_dist.append(mpps_dev_dist)
    mpps_mean_dev_dist = np.mean([np.linalg.norm(X_mpps_dist[i] - x[i]) for i in range(n)])
    mpps_mean_devs_dist.append(mpps_mean_dev_dist)
    print(f'{mpps_dev:.3f} {mpps_dev_dist:.3f} {len(cleanlog[0])}')
    results_list.append((xnorm, cvx_dev, cvx_mean_dev, mpps_dev, mpps_mean_dev, mpps_dev_dist, mpps_mean_dev_dist))

figsize = (10, 6)
plt.figure(figsize=figsize)
plt.hist([mpps_devs[i] - cvx_devs[i] for i in range(numtests)], bins=np.arange(-0.16, 0.1, 0.02))
plt.xlabel(r'$\|\hat{X} - X_0\| - \|\bar{X} - X_0\|$')
plt.ylabel('Count')
plt.axvline(x=0, color='r', linestyle='--')
plt.savefig('figs/fig_4_paired_diffs.pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=figsize)
plt.hist([mpps_mean_devs[i] - cvx_mean_devs[i] for i in range(numtests)], bins=np.arange(-0.03, 0.03, 0.003))
plt.xlabel(r'$\frac{1}{n}\sum_{i=1}^n\|\hat{X}_i - X_{0i}\| - \|\bar{X}_i - X_{0i}\|$')
plt.ylabel('Count')
plt.axvline(x=0, color='r', linestyle='--')
plt.savefig('figs/fig_4_paired_diffs_mean.pdf', bbox_inches='tight')
plt.close()

columns = ['xnorm', 'IP_Deviation', 'IP_Mean_Deviation', 'MPPS_Deviation', 'MPPS_Mean_Deviation', 'MPPS_Deviation_Dist', 'MPPS_Mean_Deviation_Dist']
def write(filename, results_list):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['s'] + columns)
        for s, values in enumerate(results_list):
            writer.writerow([s] + list(values))

 
# Get the directory of the current script
dir_path = os.path.dirname(os.path.abspath(__file__))

# Create the full path to the file
file_path = os.path.join(dir_path, "term_output.csv")
write(file_path, results_list)

bw_adjust = 0.8

def saveKDE(data, labels, name, filename, bw_adjust, colors=['blue', 'red']):
    plt.figure(figsize=figsize)
    for x, label, color in zip(data, labels, colors):
        sns.kdeplot(data=x, label=label, fill=True, color=color, bw_adjust=bw_adjust)

    plt.xlabel(name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()

    # Show the plot
    plt.savefig('figs/' + filename, bbox_inches='tight')
    plt.close()


saveKDE([mpps_devs, cvx_devs], ['Early Termination', 'Relaxation Solution'], 'Distance from True Locations', 'fig_4_early_termination_density.pdf', bw_adjust)

saveKDE([mpps_mean_devs, cvx_mean_devs], ['Early Termination', 'Relaxation Solution'], 'Mean Distance from True Locations', 'fig_4_early_termination_density_mean.pdf', bw_adjust)

saveKDE([mpps_devs_dist, cvx_devs], ['Early Termination', 'Relaxation Solution'], 'Distance from True Locations', 'fig_4_early_termination_density_dist.pdf', bw_adjust)

saveKDE([mpps_mean_devs_dist, cvx_mean_devs], ['Early Termination', 'Relaxation Solution'], 'Mean Distance from True Locations', 'fig_4_early_termination_density_mean_dist.pdf', bw_adjust)
