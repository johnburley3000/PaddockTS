#!/bin/bash

# Notes:
# I had to make my pgpass have both of these lines for the pre-segmentation to work
    # dea-db.nci.org.au:6432:datacube:cb8590:{PASS_KEY}
    # *:*:*:cb8590:{PASS_KEY}
# I had to copy sam_vit_h_4b8939.pth from John's PadSeg to my PadSeg for the SamGEO to work
# The canopy_height.download_new_tiles() needs an AWS account and a file named .aws/credentials in your home directory (e.g. /home/147/cb8590) with the 3 lines below. It runs last, so the other variables should still download even if you don't set this up.
# [default]
# aws_access_key_id = ACCESS_KEY
# aws_secret_access_key = SECRET_KEY

## Config:
# specify working directory and storage directory:
wd=/home/147/cb8590/Projects/PaddockTS  # We can't read or write to each other's home folders
dir=/g/data/xe2/cb8590/Data/PadSeg      # We can read but can't write to each other's gdata or scratch folders
tmpdir=/scratch/xe2/cb8590/tmp  

# params
stub=MILG
lat=-35.289561061551844
lon=149.06381325367872
buffer=0.005    # In degrees in a single direction. For example, 0.01 degrees is about 1km so it would give a 2kmx2km area.
start='2017-01-01'
end_='2024-12-31'

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

## Run third job (independent of first two)
job_id3=$(qsub -v wd=$wd,stub=$stub,dir=$dir,tmpdir=$tmpdir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_ Code/run_environmental.sh)
echo "Third job submitted with ID $job_id3"