#!/bin/bash

# specify working directory and storage directory:
wd=/home/147/cb8590/Projects/PaddockTS 
dir=/g/data/xe2/cb8590/Data/shelter/ 
tmpdir=/scratch/xe2/cb8590/shelter

# params
stub=JUNEE
lat=-34.893837
lon=147.594979

# stub=BOMB
# lat=-35.273807
# lon=148.272152
buffer=0.05 
start='2017-01-01'
end_='2024-12-31'

## Run first job
job_id1=$(qsub -v wd=$wd,stub=$stub,dir=$dir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_ Code/run_sentinel.sh)
echo "First job submitted with ID $job_id1"

# Run second job
job_id2=$(qsub -v wd=$wd,stub=$stub,dir=$dir,tmpdir=$tmpdir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_ Code/run_environmental.sh)
echo "Second job submitted with ID $job_id2"

# Run third job (dependent on first and second)
if [[ -z "$job_id1" || -z "$job_id2" ]]; then
    echo "Failed to submit the first or second job."
    exit 1
else
    # Specify dependency on both jobs
    job_id3=$(qsub -W depend=afterok:$job_id1:$job_id2 \
        -v wd=$wd,stub=$stub,dir=$dir,tmpdir=$tmpdir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_ \
        Code/run_shelter.sh)
    echo "Third job submitted with ID $job_id3"
fi

# job_id3=$(qsub -v wd=$wd,stub=$stub,dir=$dir,tmpdir=$tmpdir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_, Code/run_shelter.sh)
# echo "Third job submitted with ID $job_id3"