import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import os

dir_path = os.path.dirname(os.path.abspath(__file__))

# Create the full path to the file
file_path = os.path.join(dir_path, "term_output.csv")
df = pd.read_csv(file_path)

bw_adjust = 1.0

plt.figure(figsize=(10, 10))  # Set figure size

sns.kdeplot(data=df['MPPS_Mean_Deviation'], label='Early Termination', fill=True, color='blue', bw_adjust=bw_adjust)

# Plot KDE for group 2
sns.kdeplot(data=df['IP_Mean_Deviation'], label='IP Relaxation Solution', fill=True, color='red', bw_adjust=bw_adjust)

# Customize the plot
plt.xlabel('Mean Distance from True Location', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()  # Add legend to distinguish groups

# Show the plot
plt.savefig('figs/fig_4_early_termination_density_mean.pdf', bbox_inches='tight')
plt.close()




plt.figure(figsize=(10, 10))
plt.hist([mpps_devs_i - cvx_devs_i for mpps_devs_i, cvx_devs_i in zip(df['MPPS_Deviation'], df['IP_Deviation'])], bins=np.arange(-0.16, 0.1, 0.02))
plt.xlabel(r'$\|\hat{X} - X_0\| - \|\bar{X} - X_0\|$')
plt.ylabel('Count')
# vertical line at 0 in red
plt.axvline(x=0, color='r', linestyle='--')
# tight layout
plt.tight_layout()
plt.savefig('figs/fig_4_paired_diffs_early.pdf')
plt.close()

# Generate table with median performance for this noise factor and random function for MPPS and for IP

for measure in [np.median, np.mean]:
    if measure == np.median:
        print('Median')
    else:
        print('Mean')
    print('\tNormal', '\t\tUniform')
    print('\t'.join(['nf', 'IP', 'MPPS', 'IP', 'MPPS']))
    for nf in [.03, .05, .07]:
        ip_median_dev_normal = measure(df[(df['nf'] == nf) & (df['rand'] == np.random.randn.__name__)]['IP_Deviation'])
        mpps_median_dev_normal = measure(df[(df['nf'] == nf) & (df['rand'] == np.random.randn.__name__)]['MPPS_Deviation'])
        ip_median_dev_uniform = measure(df[(df['nf'] == nf) & (df['rand'] == 'uniform')]['IP_Deviation'])
        mpps_median_dev_uniform = measure(df[(df['nf'] == nf) & (df['rand'] == 'uniform')]['MPPS_Deviation'])
        print(f'{nf}\t{ip_median_dev_normal:.3f}\t{mpps_median_dev_normal:.3f}\t{ip_median_dev_uniform:.3f}\t{mpps_median_dev_uniform:.3f}')

# Export previous output as a latex table
