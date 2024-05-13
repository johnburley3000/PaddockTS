#!/bin/bash
#PBS -N PlanetXarray
#PBS -P xe2
#PBS -q express
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
python3 Code/planet_xarray_3b.py --indir /g/data/xe2/datasets/Planet/Farms/SPRV --orderid 5fb7ea5a-05ec-4784-a2c1-e4a59540f914 --outpath /home/147/cb8590/practice/SPRV_xarray_3b_400MB.pkl

# To submit
# qsub repos/PaddockTS/Code/run_planet_xarray_3b.sh