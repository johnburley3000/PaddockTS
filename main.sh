#!/bin/bash

## Config:
stub=test3
dir=/g/data/xe2/John/Data/PadSeg/
lat=-34.3890
lon=148.4695
buffer=0.01
start='2020-01-01'
end_='2020-03-31'
#
min_area_ha=10
max_area_ha=1500
max_perim_area_ratio=40


## Run first job
job_id1=$(qsub -v stub=$stub,dir=$dir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_ Code/run_pre-seg.sh)
echo "First job submitted with ID $job_id1"

## Run second job (if job ID was produced, and when job complete)
if [[ -z "$job_id1" ]]
then
    echo "Failed to submit the first job."
    exit 1
else
    echo "Submitting second job, dependent on the completion of the first."
    qsub -W depend=afterok:$job_id1 -v stub=$stub,dir=$dir,min_area_ha=$min_area_ha,max_area_ha=$max_area_ha,max_perim_area_ratio=$max_perim_area_ratio Code/run_SAMGeo_paddocks-ts.sh
    #qsub -v stub=$stub,dir=$dir,min_area_ha=$min_area_ha,max_area_ha=$max_area_ha,max_perim_area_ratio=$max_perim_area_ratio Code/run_SAMGeo_paddocks-ts.sh
fi
