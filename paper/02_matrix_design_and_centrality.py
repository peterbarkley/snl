

# %%
import cvxpy as cp
import numpy as np
from sys import path
from pathlib import Path
# Sets parent directory as a string path
parent_dir = str(Path(__file__).resolve().parent.parent)
path.append(parent_dir)
from snl import getZfromNeighbors, getBlockIncidence, getSNLProxData, generateRandomData, solve_snl_fusion
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from collections import defaultdict
from datetime import datetime
# import matplotlib
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['text.usetex'] = True
# %%

# %% [markdown]
# ## Setup
# %%

numtests = 20
n = 30
m = 6
d = 2
rd = 0.7 # neighborhood radius (sensors are in [0,1]^d), distances are only given within this radius
nf = 0.05 # noise factor. d_{ij} = d^0_{ij}(1 + nf * r_{ij}) where r_{ij} is a standard Gaussian rv
mpps_alpha = 10.0 # scaling parameter
mpps_gamma = 0.999 # step size parameter
itrs = 500

parallel = False
if parallel:
    # requires mpi and mpi4py
    from distributed_snl import distributed_block_sparse_solve
    from snl import loadLogs
else:
    # requires the development branch of oars
    from oars import solve

# %%
# %% [markdown]
# ## Functions
# %%

def solver(n, a, d, dx, aa, Ni, Na, x, Z, primal, alpha=1.0, gamma=0.5, itrs=1000, parallel=False):
    data, proxlist = getSNLProxData(n, a, d, dx, aa, Ni, Na)
    if not parallel:
        X, log, _, _ = solve(2*n, data, proxlist, Z=Z, W=Z, warmstartprimal=primal, alpha=alpha, gamma=gamma, itrs=itrs, verbose=False)
    else:
        X, _ = distributed_block_sparse_solve(2*n, data, proxlist, Z=Z, warmstartprimal=primal, alpha=alpha, gamma=gamma, itrs=itrs, verbose=False, logging=True)
        log = loadLogs(n)
    return X, log



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


# %% [markdown]
# ## Testing
# %%

print("Matrix Design Test and Centrality Test")

logs = defaultdict(list)
centers = [] 
ip_centralities = [] 
actual_centralities = [] 

def getCentrality(x, center):
    return np.mean([np.linalg.norm(pt - center) for pt in x])

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

def compare_errors_single(loglist, lognames, itrs, error_ref=None, metric=np.median, ylabel='Median MSE', reflabel='Relaxation solution'):
    
    for logs, name in zip(loglist, lognames):
        line, = plt.plot([metric([log_i[j] for log_i in logs]) for j in range(itrs)], label=name)

    # line at average cvx deviation
    if error_ref is not None:
        plt.axhline(y=error_ref, color='r', linestyle='--', label=reflabel)
    plt.legend()
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.xlabel('Iteration')

    return plt

errors_sk = []
errors_con = []
errors_resist = []
errors_slem = []
errors_centered = []

print('date', 'time', 'test', 'mosek',  'sk', 'con', 'resist', 'slem')
for s in range(numtests):
    print(datetime.now(), s, end=' ')
    a, x, da, dx, aa, Ni, Na = generateRandomData(n, m, d, rd, nf, seed=s)
    
    Mz = getBlockIncidence(Ni)
    center = np.mean(a, axis=0)
    centers.append(center)
    actual_centralities.append(getCentrality(x, center))

    # Mosek
    X_cvx, cvx_val = solve_snl_fusion(a, n, aa, dx, Ni, Na)
    # X_cvx, cvx_val = solve_snl_vec(a, n, aa, dx, Ni, Na) # CVXPY (slower, especially as size increases)
    cvx_dev = np.sum((X_cvx[d:, :d] - x)**2)/n
    print(np.round(cvx_dev,3), end=' ')
    ip_centralities.append(getCentrality(X_cvx[d:, :d], center))

    # SK
    blankprimal = np.zeros((n+d, n+d))
    blankprimal[:d, :d] = np.eye(d)
    Z = getZfromNeighbors(Ni)
    X_mpps, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Z, blankprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=parallel)
    mpps_err = np.sum((X_mpps[d:, :d] - x)**2)/n
    # cleanlog = loadLogs(n)
    print(f'{mpps_err:.3f}', end=' ')
    # logs['Sinkhorn-Knopp'].append(cleanlog)
    single_error = np.array([np.sum([((np.array(cleanlog[i+n][j][-d:]) + np.array(cleanlog[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
    errors_sk.append(single_error)

    # Center Warmstart
    primal = blankprimal.copy()
    centeredX = np.tile(center, (n, 1))
    primal[:d, d:] = centeredX.T
    primal[d:, :d] = centeredX
    primal[d:, d:] = centeredX @ centeredX.T
    X, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Z, primal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=parallel)
    logs['Centered'].append(cleanlog)
    single_error = np.array([np.sum([((np.array(cleanlog[i+n][j][-d:]) + np.array(cleanlog[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
    errors_centered.append(single_error)

    # Max Connectivity
    Zc, _ = getMaxFiedler_Fast(n, Mz)
    X, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Zc, blankprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=parallel)
    mpps_dev = np.sum((X[d:, :d] - x)**2)/n
    print(np.round(mpps_dev,3), end=' ')
    logs['Max Connectivity'].append(cleanlog)
    single_error = np.array([np.sum([((np.array(cleanlog[i+n][j][-d:]) + np.array(cleanlog[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
    errors_con.append(single_error)

    # Min Resist
    blankprimal = np.zeros((n+d, n+d))
    blankprimal[:d, :d] = np.eye(d)
    Zr, _ = getMinResist_Fast(n, Mz)
    X, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Zr, blankprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=parallel)
    mpps_dev = np.sum((X[d:, :d] - x)**2)/n
    print(np.round(mpps_dev,3), end=' ')
    logs['Min Resistance'].append(cleanlog)
    single_error = np.array([np.sum([((np.array(cleanlog[i+n][j][-d:]) + np.array(cleanlog[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
    errors_resist.append(single_error)

    # Min SLEM    
    blankprimal = np.zeros((n+d, n+d))
    blankprimal[:d, :d] = np.eye(d)
    Zs, _ = getMinSLEM_Fast(n, Mz)
    X, cleanlog = solver(n, a, d, dx, aa, Ni, Na, x, Zs, blankprimal, alpha=mpps_alpha, gamma=mpps_gamma, itrs=itrs, parallel=parallel)
    mpps_dev = np.sum((X[d:, :d] - x)**2)/n
    print(np.round(mpps_dev,3))
    logs['Min SLEM'].append(cleanlog)
    single_error = np.array([np.sum([((np.array(cleanlog[i+n][j][-d:]) + np.array(cleanlog[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
    errors_slem.append(single_error)


designs = ['Sinkhorn-Knopp', 'Max Connectivity', 'Min Resistance', 'Min SLEM']
fig = compare_errors_single([errors_sk, errors_con, errors_resist, errors_slem], designs, itrs)

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
