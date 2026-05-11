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

# Configuration variables
n = 30 # number of sensors
m = 6  # number of anchors
d = 2  # dimension of space
rd = 0.7 # neighborhood radius
nf = 0.05 # noise factor
mpps_alpha = 10.0 # scaling parameter
mpps_gamma = 0.999 # step size parameter
admm_alpha = 150.0 # ADMM scaling parameter

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

xhat = np.zeros_like(x)
for i in range(n):
    xhat[i] = cleanlog[i+n][-1][-d:]
fig = plotPoints(a, x, xhat=xhat, xbar=cvx_X_val, link=True)
fig.savefig('figs/fig_3_point_plot_dist.pdf', bbox_inches='tight')
plt.close()

fig = plotPoints(a, x, xhat=X_mpps[d:, :d], xbar=cvx_X_val, link=True)
fig.savefig('figs/fig_3_point_plot.pdf', bbox_inches='tight')
plt.close()
