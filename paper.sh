#!/bin/bash
# set -e # Exit immediately if a command exits with a non-zero status.
module load app/texlive/20220321

# Document machine that executed the code
echo "Machine: $(hostname)" >> paper_info.txt

echo "Running 00_alphas_sweep.py..." >> paper_info.txt
python -u paper/00_alphas_sweep.py >> paper_info.txt
python -u paper/00_plot_alphas.py >> paper_info.txt

echo "Running 01_mpps_admm_comparison.py..." >> paper_info.txt
python -u paper/01_mpps_admm_comparison.py >> paper_info.txt

echo "Running 02_matrix_design_and_centrality.py..." >> paper_info.txt
python -u paper/02_matrix_design_and_centrality.py >> paper_info.txt

echo "Running 03_matrix_design_time.py..." >> paper_info.txt
python -u paper/03_matrix_design_time.py >> paper_info.txt

echo "Running 04_point_plot.py..." >> paper_info.txt
python -u paper/04_point_plot.py >> paper_info.txt

echo "Running 05_termination.py..." >> paper_info.txt
python -u paper/05_termination.py >> paper_info.txt
python -u paper/05_plot_term.py >> paper_info.txt

echo "Running 06_size_comparison.py ... - this requires mpi and 251 cores. Please run on a node with at least 251 cores." >> paper_info.txt
mpiexec -n 1 python -u paper/06_size_comparison.py >> paper_info.txt
python -u paper/06_plot_size.py >> paper_info.txt


echo "All tests completed successfully."
