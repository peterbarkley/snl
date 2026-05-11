import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# 1. Load the data
df = pd.read_csv('errors_data.csv')

# 2. Group by Algorithm and Alpha, and calculate the mean for all numerical columns
df_mean = df.groupby(['Algorithm', 'Alpha']).mean().reset_index()

# Separate the dataframes by algorithm for easier plotting
mpps_data = df_mean[df_mean['Algorithm'] == 'MPPS']
admm_data = df_mean[df_mean['Algorithm'] == 'ADMM']

# 3. Create a figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: Mean Min Error vs Alpha ---
axes[0].plot(mpps_data['Alpha'], mpps_data['Min_Error'], marker='o', label='MPPS')
axes[0].plot(admm_data['Alpha'], admm_data['Min_Error'], marker='s', label='ADMM')
axes[0].set_title('Mean Min Error vs Alpha')
axes[0].set_xlabel('Alpha')
axes[0].set_ylabel('Mean Min Error')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(True)

# --- Plot 2: Mean Min Error Index vs Alpha ---
axes[1].plot(mpps_data['Alpha'], mpps_data['Min_Error_Idx'], marker='o', label='MPPS')
axes[1].plot(admm_data['Alpha'], admm_data['Min_Error_Idx'], marker='s', label='ADMM')
axes[1].set_title('Mean Min Error Index vs Alpha')
axes[1].set_xlabel('Alpha')
axes[1].set_ylabel('Mean Min Error Index (Iteration)')
axes[1].legend()
axes[1].grid(True)

# --- Plot 3: Mean Sum Error vs Alpha ---
axes[2].plot(mpps_data['Alpha'], mpps_data['Sum_Error'], marker='o', label='MPPS')
axes[2].plot(admm_data['Alpha'], admm_data['Sum_Error'], marker='s', label='ADMM')
axes[2].set_title('Mean Area Under the Curve (AUC) vs Alpha')
axes[2].set_xlabel('Alpha')
axes[2].set_ylabel('Mean AUC')
axes[2].legend()
axes[2].grid(True)

# Adjust layout to prevent overlap and display the plots
plt.tight_layout()
plt.savefig('figs/fig_0_alphas_vs_error_all_alg_log.pdf', bbox_inches='tight')
plt.show()
