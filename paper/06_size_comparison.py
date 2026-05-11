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
    # with h5py.File(filename, 'a') as f:  # 'a' mode creates the file if it doesn't exist, or appends
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
        # x_vals[n].append(x)
        # x_vals.append(x)
        

        # CVX
        # start = time()
        # X_cvx, cvx_val = solve_snl_fusion(a, n, aa, dx, Ni, Na, d)
        # mosek_time = time() - start
        # cvx_dev = np.linalg.norm(X_cvx[d:, :d] - x, 'fro')
        # xnorm = np.linalg.norm(x)
        # cvx_error = cvx_dev/xnorm
        # print(f'{cvx_error:.3f} {mosek_time:.3f}', end=' ')
        # cvx_vals[n].append(cvx_val)
        # cvx_devs[n].append(cvx_dev)
        # cvx_X_vals[n].append(X_cvx[d:, :d])
        # cvx_errors[n].append(cvx_error)

        # OARS setup
        blankprimal = np.zeros((n+d, n+d))
        blankprimal[:d, :d] = np.eye(d)

        alpha = 10.0
        oarsgamma = .95
        # print(datetime.now(), 'getting Z')
        Z = getZfromNeighbors(Ni)

        # print(datetime.now(), 'getting data')
        data, proxlist = getSNLProxData(n, a, d, dx, aa, Ni, Na)
        X_oars, variance, sum_gradient, oars_log, oars_time, itr, update_norm = distributed_block_sparse_solve(2*n, data, proxlist, Z=Z, warmstartprimal=blankprimal.copy(), alpha=alpha, gamma=oarsgamma, itrs=itrs, verbose=False, logging=True)
        # if itr+1 < itrs:
        #     print(datetime.now(), 'Early stop at iteration', itr)
        # print(datetime.now(), 'Update norm', update_norm)
        # tolerances.append([s, variance, sum_gradient])
        # variances.append(variance)
        # sum_gradients.append(variance)
        # oars_dev = np.linalg.norm(X_oars[d:, :d] - x, 'fro')
        cleanlog = loadLogs(n)
        # oarsval = sum(cleanlog[i][itrs-1][3] for i in range(n))
        # oars_devs[n].append(oars_dev)
        # logs[('MPPS', n)].append(cleanlog)
        # averaged_oars_error = oars_dev/xnorm
        single_oars_error = np.array([np.sum([((np.array(cleanlog[i+n][j][-d:]) + np.array(cleanlog[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(cleanlog[0]))]) / n
        # errors[('MPPS', n)].append(single_oars_error)
        # times[('MPPS', n)].append(oars_time)
        # mppstimes = np.array([cleanlog[-1][j][0] - cleanlog[-1][1][0] for j in range(1,len(cleanlog[0]))])
        
        save_experiment_data(f, size=n, seed=i, algorithm="mpps", log=cleanlog, x=x, d=d)
        # alltimes[('MPPS', n)].append(mppstimes)
        # print(single_oars_error)
        print(f'{min(single_oars_error):.3e} {oars_time:.3f}', end=' ')

        admm_alpha = 150.0 # ADMM scaling parameter
        admm_itrs = 4*itrs
        mean_X_admm, admm_time, update_norm, itr = solve_admm_dist(a, n, dx, aa, Ni, Na, warmstartprimal=blankprimal, alpha=admm_alpha, itrs=admm_itrs, logging=True)
        # if itr+1 < admm_itrs:
        #     print(datetime.now(), 'Early stop at iteration', itr)
        # print(datetime.now(), 'Update norm', update_norm)
        admm_log = loadLogs(n)
        # logs[('ADMM', n)].append(cleanlog)
        # dev = np.linalg.norm(mean_X_admm[d:, :d] - x, 'fro')
        # mean_dev = np.mean([np.linalg.norm(mean_X_admm[i+d, :d] - x[i]) for i in range(n)])
        # logs['ADMM'].append(admm_log)
        single_error = np.array([np.sum([((np.array(admm_log[i+n][j][-d:]) + np.array(admm_log[i][j][-d:]) )/2 - x[i])**2 for i in range(n)]) for j in range(len(admm_log[0]))])/n
        # admmtimes = np.array([admm_log[-1][j][0] - admm_log[-1][1][0] for j in range(1,len(admm_log[0]))])
        # errors[('ADMM', n)].append(single_error)
        
        # alltimes[('ADMM', n)].append(admmtimes)
        # times[('ADMM', n)].append(admm_time)
        print(f'{min(single_error):.3e} {admm_time:.3f}')
        save_experiment_data(f, size=n, seed=i, algorithm="admm", log=admm_log, x=x, d=d)
        s += 1
    # fig = compare_errors(errors, errors.keys()) #[x_vals[n] for n in sizes[:size_idx+1]])

    # # save plot as pdf
    # fig.savefig('size_comparison_errors_median.pdf', bbox_inches='tight')
    # fig.close()
    # fig = compare_errors_time(errors, alltimes, errors.keys(), itrs) #[x_vals[n] for n in sizes[:size_idx+1]])

    # # save plot as pdf
    # fig.savefig('size_comparison_errors_median_time.pdf', bbox_inches='tight')
    # fig.close()
    # fig = compare_errors([oarslogs[n] for n in sizes[:size_idx+1]], sizes[:size_idx+1], numtests, itrs, x_vals, error=mean_rel_error)

    # # save plot as pdf
    # fig.savefig('size_comparison_mean_errors_median.pdf', bbox_inches='tight')
    # fig.close()

f.close()
