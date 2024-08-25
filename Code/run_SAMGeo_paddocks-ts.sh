#!/bin/bash
#PBS -N SAMGeo
#PBS -l mem=128GB
#PBS -l jobfs=24GB
#PBS -q gpuvolta
#PBS -l ncpus=12,ngpus=1
#PBS -P xe2
#PBS -l walltime=04:00:00
#PBS -l storage=gdata/xe2

# Print out input variables to the error log
echo "Running SamGeo script and extract paddock ts script using the following input variables:"
echo "stub: $stub"
echo "outdir: $dir"
echo "min_area_ha: $min_area_ha"
echo "max_area_ha: $max_area_ha"
echo "max_perim_area_ratio: $max_perim_area_ratio"
###

#cd /home/106/jb5097/Projects/PaddockTS
cd $wd

## run it like:
# qsub -v stub=<>,dir=<>,min_area_ha=<>,max_area_ha=<>,max_perim_area_ratio=<> Code/run_SAMGeo_paddocks-ts.sh

# activate virtual environment:
source /g/data/xe2/John/geospatenv/bin/activate

# Run script:
python Code/02_SAMGeo_paddocks.py $stub $dir --min_area_ha $max_perim_area_ratio --max_area_ha $max_area_ha --max_perim_area_ratio $max_perim_area_ratio

# stub: unique id that specifies the input image and sets output name
# minimum area for the resulting polygons
# maximum perimeter:area ratio for resulting polygons

# Now run the code that extracts paddock-level reflectance from Sentinel and saves the paddock-variable-time (pvt) array
python Code/03_paddock-ts.py --stub $stub --outdir $dir 
# this assumes that the <stub>_ds2.pkl exists 


