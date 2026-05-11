# snl
Sensor Network Localization using a Decentralized Proximal Splitting Method

To run, you will need to clone the development branch of the OARS repository from https://github.com/peterbarkley/oars.git

```bash
git clone --branch development https://github.com/peterbarkley/oars.git
```

Then install OARS by running the following command:

```bash
cd oars
pip install -e .
```

Then you can install the remainder of the project's dependencies by running the following command:

```bash
pip install -r requirements.txt
```

I used the MOSEK Fusion API for reference solutions - you can choose another solver instead by commenting out `solve_snl_fusion` and uncommenting `solve_snl_vec` in the python scripts.

You will need to have latex installed in order to write the charts are they are given in the paper. Other you can just comment out the `plt.rcParams['text.usetex'] = True` line in each of the python scripts.

If you want to run the size comparison, you will need mpi installed, and you will need to run the script on a node with at least 251 cores, or reduce the maximum size in the script.

Finally, you can run the project by running the following command: 

```bash
./paper.sh
```

# Data Files
The 00_alpha_sweep script generates the errors_data.csv file. This can be plotted with 00_plot_alphas.py
The 01_mpps_admm_comparison script generates the mpps_admm_comparison.csv file.
The 03_matrix_design_time script generates the design_time.csv file. 
The 05_termination script generates the term_output.csv file. This can be plotted with 05_plot_term.py
The 06_size_comparison script generates the experiment_results.h5 file. This can be plotted with 06_plot_size.py

# Charts
00_plot_alphas.py generates fig_0_alphas_vs_error_all_alg_log.pdf.
01_mpps_admm_comparison.py generates fig_1_mpps_admm_comparison.pdf and fig_1_mpps_admm_comparison_warm.pdf. 
02_matrix_design_and_centrality.py generates fig_2_sk_oars_error.pdf and fig_3_centroid_distance.pdf.
03_matrix_design_time.py generates fig_2_design_time.pdf.
04_point_plot.py generates fig_3_point_plot_dist.pdf.
05_plot_term.py generates fig_4_paired_diffs_mean.pdf and fig_4_early_termination_density_mean.pdf.
06_plot_size.py generates size_comparison_grid_mean.pdf and size_grid_times_mean.pdf.
