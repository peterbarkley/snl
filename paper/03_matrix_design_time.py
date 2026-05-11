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

# Save as csv
df.to_csv('design_time.csv')

# Plot the mean time for each method as a function of n
plt.plot(df.groupby('Nodes').mean())
plt.ylim(-0.5, 10.5)
plt.xlabel('Nodes')
plt.ylabel('Time (s)')
plt.legend(algnames)
plt.savefig('figs/fig_2_design_time.pdf')
plt.close()
