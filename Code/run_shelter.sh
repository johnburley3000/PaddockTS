#!/bin/bash
#PBS -N Shelter
#PBS -q normal
#PBS -l mem=128GB
#PBS -l jobfs=0GB
#PBS -l ncpus=12
#PBS -P xe2
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/xe2+gdata/ub8

# Print out input variables to the error log
echo "Running shelter script with the following variables:"
echo "stub: $stub"
echo "outdir: $dir"
echo "tmpdir: $tmpdir"
echo "latitude: $lat"
echo "longitude: $lon"
echo "buffer: $buffer"
###


# activate virtual environment:
source /g/data/xe2/datasets/Environments/paddockenv3_11/bin/activate
module load python3/3.11.7 

cd $wd
echo "Working directory: $wd"

# Run script:
python Code/05_shelter.py --stub $stub --outdir $dir --tmpdir $tmpdir --lat $lat --lon $lon --buffer $buffer --start_time $start_time --end_time $end_time

# Example
# python Code/05_shelter.py --stub BOMB --outdir /g/data/xe2/cb8590/Data/shelter/ --tmpdir /scratch/xe2/cb8590/shelter --lat -35.273807 --lon 148.272152 --buffer 0.05 --start_time '2017-01-01' --end_time '2024-12-31'