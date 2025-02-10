#!/bin/bash

# specify working directory and storage directory:
wd=/home/147/cb8590/Projects/PaddockTS 
dir=/g/data/xe2/cb8590/Data/shelter/ 
tmpdir=/scratch/xe2/cb8590/shelter

# params
stub=songyuan
lat=45.15
lon=125.00

buffer=0.05 
start='2017-01-01'
end_='2024-12-31'

# params for paddock filtering
min_area_ha=10
max_area_ha=1500
max_perim_area_ratio=40

# ## Run first job
job_id1=$(qsub -v wd=$wd,stub=$stub,dir=$dir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_ Code/run_sentinel.sh)
echo "First job submitted with ID $job_id1"

## Run second job (after first job complete)
if [[ -z "$job_id1" ]]
then
    echo "Failed to submit the first job."
    exit 1
else
    echo "Submitting second job, dependent on the completion of the first."
    job_id2=$(qsub -W depend=afterok:$job_id1 -v wd=$wd,stub=$stub,dir=$dir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_ Code/run_pre-seg.sh)
    # qsub -W depend=afterok:$job_id2 -v wd=$wd,stub=$stub,dir=$dir,min_area_ha=$min_area_ha,max_area_ha=$max_area_ha,max_perim_area_ratio=$max_perim_area_ratio Code/run_SAMGeo_paddocks-ts.sh
    #qsub -v stub=$stub,dir=$dir,min_area_ha=$min_area_ha,max_area_ha=$max_area_ha,max_perim_area_ratio=$max_perim_area_ratio Code/run_SAMGeo_paddocks-ts.sh
fi
# qsub -v wd=$wd,stub=$stub,dir=$dir,min_area_ha=$min_area_ha,max_area_ha=$max_area_ha,max_perim_area_ratio=$max_perim_area_ratio Code/run_SAMGeo_paddocks-ts.sh

# ## Run third job (if job ID was produced, and when job complete)
if [[ -z "$job_id2" ]]
then
    echo "Failed to submit the second job."
    exit 1
else
    echo "Submitting third job, dependent on the completion of the second."
    qsub -W depend=afterok:$job_id2 -v wd=$wd,stub=$stub,dir=$dir,min_area_ha=$min_area_ha,max_area_ha=$max_area_ha,max_perim_area_ratio=$max_perim_area_ratio Code/run_SAMGeo_paddocks-ts.sh
    #qsub -v stub=$stub,dir=$dir,min_area_ha=$min_area_ha,max_area_ha=$max_area_ha,max_perim_area_ratio=$max_perim_area_ratio Code/run_SAMGeo_paddocks-ts.sh
fi
# qsub -v wd=$wd,stub=$stub,dir=$dir,min_area_ha=$min_area_ha,max_area_ha=$max_area_ha,max_perim_area_ratio=$max_perim_area_ratio Code/run_SAMGeo_paddocks-ts.sh
# echo "SAMGeo job submitted"


# # Run fourth job
job_id4=$(qsub -v wd=$wd,stub=$stub,dir=$dir,tmpdir=$tmpdir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_ Code/run_environmental.sh)
echo "Fourth job submitted with ID $job_id4"

# # Run fifth job (dependent on first and second)
# if [[ -z "$job_id1" || -z "$job_id2" ]]; then
#     echo "Failed to submit the first or second job."
#     exit 1
# else
#     # Specify dependency on both jobs
#     job_id3=$(qsub -W depend=afterok:$job_id1:$job_id2 \
#         -v wd=$wd,stub=$stub,dir=$dir,tmpdir=$tmpdir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_ \
#         Code/run_shelter.sh)
#     echo "Third job submitted with ID $job_id3"
# fi

# job_id3=$(qsub -v wd=$wd,stub=$stub,dir=$dir,tmpdir=$tmpdir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_, Code/run_shelter.sh)
# echo "Third job submitted with ID $job_id3"
