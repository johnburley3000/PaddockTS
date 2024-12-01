#!/bin/bash
#PBS -N Environmental
#PBS -q copyq
#PBS -l mem=32GB
#PBS -l ncpus=1
#PBS -P xe2
#PBS -l walltime=00:30:00
#PBS -l storage=gdata/xe2+gdata/ub8
#PBS -l jobfs=24GB


# Notes:
# 'copyq' queue required for internet access
# 'ub8' storage access required for OzWald soil moisture 

# Print out input variables to the error log
echo "Running environmental script with the following variables:"
echo "stub: $stub"
echo "outdir: $dir"
echo "tmpdir: $tmpdir"
echo "latitude: $lat"
echo "longitude: $lon"
echo "buffer: $buffer"
###


# activate virtual environment:
source /g/data/xe2/John/geospatenv/bin/activate
module load gdal/3.6.4
# module load netcdf/4.7.3

cd $wd
echo "Working directory: $wd"

# Run script:
# Actually don't need the lat, lon, buffer, start_time, or end_time. Just there for easier copy pasting
python Code/04_environmental.py --stub $stub --outdir $dir --tmpdir $tmpdir --lat $lat --lon $lon --buffer $buffer --start_time $start_time --end_time $end_time


