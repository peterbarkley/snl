import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
import os

dir_path = os.path.dirname(os.path.abspath(__file__))

# Create the full path to the file
file_path = os.path.join(dir_path, "term_output.csv")
df = pd.read_csv(file_path)

bw_adjust = 1.0
size = 5
plt.figure(figsize=(size, size))  # Set figure size

sns.kdeplot(data=df['MPPS_MSE'], label='Early Termination', fill=True, color='blue', bw_adjust=bw_adjust)

# Plot KDE
sns.kdeplot(data=df['IP_MSE'], label='IP Relaxation Solution', fill=True, color='red', bw_adjust=bw_adjust)

# Customize the plot
plt.xlabel(r'$\frac{1}{n}\|\hat{X} - X_0\|^2_F$', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()  # Add legend to distinguish groups

# Show the plot
plt.savefig('figs/fig_4_early_termination_density_mean.pdf', bbox_inches='tight')
plt.close()




plt.figure(figsize=(size, size))
plt.hist([mpps_devs_i - cvx_devs_i for mpps_devs_i, cvx_devs_i in zip(df['MPPS_MSE'], df['IP_MSE'])], bins=np.arange(-0.004, 0.006, 0.001)) #
plt.xlabel(r'$\frac{1}{n}\|\hat{X} - X_0\|^2_F - \frac{1}{n}\|\bar{X} - X_0\|^2_F$')
plt.ylabel('Count')

plt.axvline(x=0, color='r', linestyle='--')

plt.tight_layout()
plt.savefig('figs/fig_4_paired_diffs_mean.pdf')
plt.close()

# Generate table with median performance for this noise factor and random function for MPPS and for IP

for measure in [np.median, np.mean]:
    if measure == np.median:
        print('Median')
    else:
        print('Mean')
    print('\tNormal', '\t\tUniform')
    print('\t&\t'.join(['nf', 'IP', 'MPPS', 'IP', 'MPPS']))
    for nf in [.03, .05, .07]:
        ip_median_dev_normal = measure(df[(df['nf'] == nf) & (df['rand'] == np.random.randn.__name__)]['IP_MSE'])
        mpps_median_dev_normal = measure(df[(df['nf'] == nf) & (df['rand'] == np.random.randn.__name__)]['MPPS_MSE'])
        ip_median_dev_uniform = measure(df[(df['nf'] == nf) & (df['rand'] == 'uniform')]['IP_MSE'])
        mpps_median_dev_uniform = measure(df[(df['nf'] == nf) & (df['rand'] == 'uniform')]['MPPS_MSE'])
        print(f'{nf} & \t\t{ip_median_dev_normal:.3e} & \t{mpps_median_dev_normal:.3e} & \t{ip_median_dev_uniform:.3e} &\t{mpps_median_dev_uniform:.3e}')

