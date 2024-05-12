#!/bin/bash
#PBS -N PlanetXarray
#PBS -P xe2
#PBS -l mem=32GB
#PBS -l jobfs=3GB
#PBS -l ncpus=1
#PBS -l walltime=00:30:00
#PBS -l storage=gdata/xe2

## Config:
wd=/home/147/cb8590/repos/PaddockTS
cd $wd

# Activate virtual environment:
source /g/data/xe2/John/geospatenv/bin/activate

# Run python script 
# python3 Code/planet_xarray_3b.py --indir /home/147/cb8590/practice/MULL --orderids 79f404e3-6b72-43fa-ac13-1b33d0afa755 --outpath /home/147/cb8590/practice/MULL_xarray_3b.pkl
python3 Code/planet_xarray_3b.py --indir /g/data/xe2/datasets/Planet/Farms/ARBO --orderids 74dbb359-bf0e-4445-a643-7423ef87edf7 --outpath /home/147/cb8590/practice/ARBO_xarray_3b_1GB.pkl

# To submit
# qsub repos/PaddockTS/Code/run_planet_xarray_3b.sh 