#!/bin/bash

# specify working directory and storage directory:
wd=/home/147/cb8590/Projects/PaddockTS 
dir=/g/data/xe2/cb8590/Data/shelter/ 
tmpdir=/scratch/xe2/cb8590/shelter  

# params
buffer=0.05 
start='2017-01-01'
end_='2024-12-31'

# Automatically submit a whole bunch of jobs for lots of locations spaced 10km apart
coordinates_file="coordinates.txt"

# Start and end coordinates
lon_start=148
lon_end=149
lat_start=-34.8
lat_end=-33.8

# Spacing
spacing=0.1

# Remove existing file if it exists
if [[ -f $coordinates_file ]]; then
    rm $coordinates_file
fi

# Generate the coordinates
for lon in $(seq $lon_start $spacing $lon_end); do
    for lat in $(seq $lat_start $spacing $lat_end); do
        echo "$lon, $lat" >> $coordinates_file
    done
done

echo "Coordinates saved to $coordinates_file"


# Loop through each coordinate in the file
while IFS=, read -r lon lat; do
    lon=$(echo $lon | xargs)
    lat=$(echo $lat | xargs)
    stub="$(printf "%.1f_%.1f" $lat $lon | sed 's/-//' | tr '.' '_')"
    # echo $stub
    
    ## Run first job
    job_id1=$(qsub -v wd=$wd,stub=$stub,dir=$dir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_ Code/run_sentinel.sh)
    echo "First job submitted with ID $job_id1"

    # # Run second job
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

done < "$coordinates_file"
