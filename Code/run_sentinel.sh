#!/bin/bash
#PBS -N Sentinel
#PBS -l mem=96GB
#PBS -l ncpus=24
#PBS -l jobfs=24GB
#PBS -P xe2
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/xe2+gdata/v10+gdata/ka08
#PBS -q normal

# Print out input variables to the error log
echo "Running with the following input variables:"
echo "stub: $stub"
echo "outdir: $dir"
echo "latitude: $lat"
echo "longitude: $lon"
echo "buffer: $buffer"
echo "start date: $start_time"
echo "end date: $end_time"
echo "-------------------"

# Requirements:
# Needs access to project v10 to load the dea modules
# (also ka08 and xe2)

#cd /home/106/jb5097/Projects/PaddockTS
cd $wd

# Setup DEA environment modules for running the Python script
module use /g/data/v10/public/modules/modulefiles
module load dea/20231204

# Run the Python script with required parameters (provide these as command-line arguments or modify the script to set defaults)
#python3 Code/00_sentinel.py --stub $1 --outdir /g/data/xe2/John/Data/PadSeg/ --lat $2 --lon $3 --buffer $4 --start_time $5 --end_time $6
python3 Code/00_sentinel.py --stub $stub --outdir $dir --lat $lat --lon $lon --buffer $buffer --start_time $start_time --end_time $end_time

# Example usage:
# python Code/05_shelter.py --stub 34_8_148_0 --outdir /g/data/xe2/cb8590/Data/shelter/ --tmpdir /scratch/xe2/cb8590/shelter --lat -35.273807 --lon 148.272152 --buffer 0.05 --start_time '2017-01-01' --end_time '2024-12-31'

