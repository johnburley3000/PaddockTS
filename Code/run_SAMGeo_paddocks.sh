#!/bin/bash
#PBS -N SAMGeo
#PBS -l ncpus=1
#PBS -l mem=20GB
#PBS -l jobfs=6GB
#PBS -q normal
#PBS -P xe2
#PBS -l walltime=04:00:00
#PBS -l storage=gdata/xe2

# Should be using a GPU for this program...

cd /home/106/jb5097/Projects/PaddockTS

## run it like:
# qsub -v stub=<>,path=<> Code/run_SAMGeo_paddocks.sh

# activate virtual environment:
source /g/data/xe2/John/geospatenv/bin/activate

# Run script:
python Code/SAMGeo_paddocks.py $stub /g/data/xe2/John/Data/PadSeg/ --min_area_ha 10 --max_area_ha 1500 --max_perim_area_ratio 30

# stub: unique id that specifies the input image and sets output name
# minimum area for the resulting polygons
# maximum perimeter:area ratio for resulting polygons


