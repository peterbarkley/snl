# %%
import numpy as np
from sys import path
from pathlib import Path
# Sets parent directory as a string path
parent_dir = str(Path(__file__).resolve().parent.parent)
path.append(parent_dir)
# path.append('/home/peter.barkley/code/parallel_sdp/code')
from snl import generateRandomData, solve_snl_fusion, getZfromNeighbors, getSNLProxData, loadLogs
from distributed_snl import distributed_block_sparse_solve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from collections import defaultdict
from datetime import datetime
plt.rcParams.update({'font.size': 6})
from time import time
from distributed_admm_gather import solve_admm_dist
import h5py

def save_experiment_data(f, size, seed, algorithm, log, x, d):
    """
    Extracts and saves experiment data to an HDF5 file.
    """
    num_functions = size
    num_iterations = len(log[0])
    
    # Pre-allocate numpy arrays for performance
    timestamps = np.zeros((num_iterations-1))
    locations = np.zeros((num_iterations-1, num_functions, d))
    
    # Extract data from the list of lists
    for itr in range(1, num_iterations):    
        timestamps[itr-1] = log[-1][itr][0] - log[-1][1][0]
        # print(timestamps)
        for i in range(num_functions):
            locations[itr-1, i, :] = (np.array(log[i][itr][-d:]) + np.array(log[i+n][itr][-d:]))/2
            
    # Save to HDF5 file
    base_path = f"size_{size}/seed_{seed}"
    
    # Save true locations (x) only once per size/seed pair
    true_loc_path = f"{base_path}/true_locations"
    if true_loc_path not in f:
        f.create_dataset(true_loc_path, data=x)
        
    # Create group for the specific algorithm
    alg_path = f"{base_path}/{algorithm}"
    
    # If rerunning an experiment, overwrite the old data
    if alg_path in f:
        del f[alg_path]
        
    # Save timestamps and locations as datasets
    f.create_dataset(f"{alg_path}/times", data=timestamps)
    f.create_dataset(f"{alg_path}/locations", data=locations)

# %%
sizes = [50, 100, 150, 200, 250]
d = 2

# %%
itrs = 500
numtests = 10
logs = defaultdict(list)
errors = defaultdict(list)

cvx_vals = defaultdict(list)
cvx_devs = defaultdict(list)
cvx_errors = defaultdict(list)
# x_vals = [] #defaultdict(list)
cvx_X_vals = defaultdict(list)
times = defaultdict(list)
alltimes = defaultdict(list)
# admm_times = 
tolerances = []
variances = []
sum_gradients = []
print(datetime.now(), 'n', 's', 'mpps error', 'mpps time', 'admm error', 'admm time')

f = h5py.File("experiment_results.h5", 'a')
s = 0
for size_idx, n in enumerate(sizes):
    m = 2*int(n**0.5)
    for i in range(numtests):
        
        print(datetime.now(), n, i, end=' ')
        a, x, da, dx, aa, Ni, Na = generateRandomData(n, m, d, 0.7, .05, seed=s)

        # OARS setup
        blankprimal = np.zeros((n+d, n+d))
        blankprimal[:d, :d] = np.eye(d)

        alpha = 10.0
        oarsgamma = .95
        Z = getZfromNeighbors(Ni)
        data, proxlist = getSNLProxData(n, a, d, dx, aa, Ni, Na)
        X_oars, variance, sum_gradient, oars_log, oars_time, itr, update_norm = distributed_block_sparse_solve(2*n, data, proxlist, Z=Z, warmstartprimal=blankprimal.copy(), alpha=alpha, gamma=oarsgamma, itrs=itrs, verbose=False, logging=True)

        cleanlog = loadLogs(n)
        single_oars_error = np.array([np.sum([((np.array(cleanlog[i+n][j][-d:]) + np.array(cleanlog[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n  
        save_experiment_data(f, size=n, seed=i, algorithm="mpps", log=cleanlog, x=x, d=d)
        print(f'{min(single_oars_error):.3e} {oars_time:.3f}', end=' ')

        admm_alpha = 150.0 # ADMM scaling parameter
        admm_itrs = 4*itrs
        mean_X_admm, admm_time, update_norm, itr = solve_admm_dist(a, n, dx, aa, Ni, Na, warmstartprimal=blankprimal, alpha=admm_alpha, itrs=admm_itrs, logging=True)
        admm_log = loadLogs(n)
        single_error = np.array([np.sum([((np.array(admm_log[i+n][j][-d:]) + np.array(admm_log[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(admm_log[0]))])/n
        print(f'{min(single_error):.3e} {admm_time:.3f}')
        save_experiment_data(f, size=n, seed=i, algorithm="admm", log=admm_log, x=x, d=d)
        s += 1

f.close()
