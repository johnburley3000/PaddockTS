#!/bin/bash
#PBS -N Download_S1
#PBS -l mem=128GB
#PBS -l jobfs=24GB
#PBS -q copyq
#PBS -l ncpus=1
#PBS -P xe2
#PBS -l walltime=08:00:00
#PBS -l storage=gdata/xe2

# Print out input variables to the error log
echo "Running Sentinel-1 data download script using the following input variables:"
echo "stub: $stub"
echo "outdir: $dir"

# Change directory to the working directory
cd $PBS_O_WORKDIR

# Activate the virtual environment
source /g/data/xe2/John/geospatenv/bin/activate

# Run the Python script to download Sentinel-1 data
python Code/download_S1.py $stub $dir

# Example:
# python Code/download_S1.py MILG_b033_2017-24 /g/data/xe2/John/Data/PadSeg
# ## run it like:
# qsub -v stub=MILG_b033_2017-24,dir=/g/data/xe2/John/Data/PadSeg Code/run_download_S1.sh
