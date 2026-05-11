# %%
import numpy as np
import h5py

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


# Adds the parent directory to the search path

from sys import path
from pathlib import Path
# Sets parent directory as a string path
parent_dir = str(Path(__file__).resolve().parent.parent)
path.append(parent_dir)

from snl import generateRandomData

# %%
def load_and_compute_error(f, size, seed, algorithm):
    """
    Loads data for a specific run and computes the average error per iteration.
    """
    base_path = f"size_{size}/seed_{seed}"
    
    # Load the data arrays into memory
    x = f[f"{base_path}/true_locations"][:]
    locations = f[f"{base_path}/{algorithm}/locations"][:]
    times = f[f"{base_path}/{algorithm}/times"][:]
        
    n = len(x)
    
    # Compute squared errors
    # shape: (n, num_iterations, d)
    diff = locations - x
    squared_errors = diff ** 2
    infty_errors = np.max(np.abs(diff), axis=(1,2))
    itr_diff = np.max(np.abs(locations[:-1] - locations[1:]), axis=(1,2))
    
    # Sum over nodes (axis 0) and dimensions (axis 2), then divide by n
    # The result is a 1D array of length `num_iterations`
    single_error = np.sum(squared_errors, axis=(1,2)) / n
    
    # Return a representative timestamp array (e.g., max time across functions per iteration)
    # or just return the whole timestamps array depending on your plotting needs
    # max_timestamps = np.max(timestamps, axis=0) 
    
    return single_error, infty_errors, itr_diff, times


# %%
def load_all_experiments(filename, sizes, seeds, algorithms):
    """
    Loads all experiment configurations and stores them in a nested dictionary.
    Structure: results[size][algorithm][seed] = {'errors': [...], 'times': [...]}
    """
    results = {}
    true_locations = {}

    
    with h5py.File(filename, 'r') as f:
        for size in sizes:
            true_locations[size] = {}
            for seed in seeds:
                true_locations[size][seed] = f[f"size_{size}/seed_{seed}/true_locations"][:]
            results[size] = {}
            for algo in algorithms:
                results[size][algo] = {}
                for seed in seeds:
                    # Check if this specific run exists in the file to avoid crash on missing data
                    base_path = f"size_{size}/seed_{seed}/{algo}"
                    if base_path in f:
                        # We can use the previously defined function here. 
                        # Note: if load_and_compute_error opens the file itself, 
                        # you might want to adjust it to accept the open file object `f` instead of `filename` 
                        # to avoid opening/closing the file dozens of times.
                        errors, inf_errors, itr_diff, times = load_and_compute_error(f, size, seed, algo)
                        
                        results[size][algo][seed] = {
                            'errors': errors,
                            'inf_errors': inf_errors,
                            'itr_diff': itr_diff,
                            'times': times
                        }
                    else:
                        print(f"Warning: Data for Size {size}, Seed {seed}, Algo {algo} not found.")
                        
    return results, true_locations

    

# %%
filename = "experiment_results.h5"
sizes = [50, 100, 150, 200, 250]
seeds = list(range(10))
algorithms = ['mpps', 'admm']

# Load everything into memory
all_results, true_locs = load_all_experiments(filename, sizes, seeds, algorithms)

from matplotlib.ticker import LogLocator
def plotAllReduced(all_results, sizes, colors = colors, measure=np.median, background=False, field='errors', xaxis='Iterations', ylabel='Median Squared Error', title='Convergence by Iteration', itrs=-1):

    # =====================================================================
    # Figure 1: Iterations vs. Median Error
    # =====================================================================
    fig1, axes1 = plt.subplots(len(sizes), figsize=(5, len(sizes)*2))

    # Ensure axes1 is iterable even if there's only one size
    if len(sizes) == 1:
        axes1 = [axes1]

    idx = 0
    for idx, size in enumerate(sizes):
        ax = axes1[idx]
        
        for algo in algorithms:
            algo_errors = []
            algo_times = []
            for seed in all_results[size][algo]:
                algo_errors.append(all_results[size][algo][seed][field][:itrs])
            
                times = all_results[size][algo][seed]['times'][:itrs]
            
                # Align the arrays: Drop the first error since time excludes iteration 0
                if len(algo_errors) > len(times):
                    algo_errors = algo_errors[1:]
                    
                algo_times.append(times)
            if algo_errors:
                # Calculate median error across all seeds
                errors = measure(algo_errors, axis=0)
                if xaxis == 'Iterations':
                    xdata = np.arange(len(errors))
                elif xaxis == 'Time (s)':
                    xdata = np.median(algo_times, axis=0)
                
                ax.plot(xdata, errors, color=colors[algo], label=algo.upper())
            if background:
                for seed in all_results[size][algo]:
                    if xaxis == 'Iterations':
                        xdata = np.arange(len(algo_errors[seed]))
                    else:
                        xdata = algo_times[seed]
                    ax.plot(xdata, algo_errors[seed], color=colors[algo], alpha=.2)

        ax.set_title(f'Problem Size: {size}')
        if idx == len(sizes)-1:
            ax.set_xlabel(xaxis)
        if idx == len(sizes)//2:
            ax.set_ylabel(ylabel)
        ax.set_yscale('log')
        minor_locator = LogLocator(base=10.0, subs=(2.0, 5.0))
        ax.xaxis.set_minor_locator(minor_locator)

        ax.yaxis.set_minor_locator(minor_locator)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        if idx == 0:
            ax.legend()

    fig1.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig1



# %%
fig = plotAllReduced(all_results, sizes, measure=np.mean, ylabel='Mean Squared Error', background=True, itrs=1500)
plt.savefig('figs/06_size_comparison_grid_mean.pdf')
plt.show()

fig = plotAllReduced(all_results, sizes, measure=np.mean, ylabel='Mean Squared Error', xaxis='Time (s)', background=True, itrs=1500, title='Convergence over Time')
plt.savefig('figs/06_size_comparison_grid_times_mean.pdf')
plt.show()
