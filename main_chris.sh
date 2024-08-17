#!/bin/bash

# Notes:
# I had to make my pgpass have both of these lines for the pre-segmentation to work
    # dea-db.nci.org.au:6432:datacube:cb8590:{passkey}
    # *:*:*:cb8590:{passkey}
# I had to copy sam_vit_h_4b8939.pth from John's PadSeg to my PadSeg for the SamGEO to work

## Config:
# specify working directory and storage directory:
wd=/home/147/cb8590/Projects/PaddockTS  # We can't read or write to each other's home folders
dir=/g/data/xe2/Chris/Data/PadSeg/      # We can read but can't write to each other's gdata folders

# params
stub=MILG_1km
lat=-35.289561061551844
lon=149.06381325367872
buffer=0.005 # this distance in all directions from (lat,lon). 0.01 degrees is ~1km in each direction which would mean 2kmx2km total
start='2020-01-01'
end_='2020-12-01'

# params for paddock filtering
min_area_ha=10
max_area_ha=1500
max_perim_area_ratio=40


## Run first job
job_id1=$(qsub -v wd=$wd,stub=$stub,dir=$dir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_ Code/run_pre-seg.sh)
echo "First job submitted with ID $job_id1"

## Run second job (if job ID was produced, and when job complete)
if [[ -z "$job_id1" ]]
then
    echo "Failed to submit the first job."
    exit 1
else
    echo "Submitting second job, dependent on the completion of the first."
    qsub -W depend=afterok:$job_id1 -v wd=$wd,stub=$stub,dir=$dir,min_area_ha=$min_area_ha,max_area_ha=$max_area_ha,max_perim_area_ratio=$max_perim_area_ratio Code/run_SAMGeo_paddocks-ts.sh
    #qsub -v stub=$stub,dir=$dir,min_area_ha=$min_area_ha,max_area_ha=$max_area_ha,max_perim_area_ratio=$max_perim_area_ratio Code/run_SAMGeo_paddocks-ts.sh
fi
