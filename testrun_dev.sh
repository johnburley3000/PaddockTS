#!/bin/bash
# This script runs a development version of the PaddockTS program
# Set the core parameters and then run the python scripts in serial. 
t_start=$(date +%s)

# when testing, start up an interactive session with small resources:
# qsub -I -P xe2 -q normal -l ncpus=4,mem=24GB,jobfs=2GB,walltime=5:00:00,wd,storage=scratch/xe2+gdata/xe2+gdata/v10+gdata/ka08

####
## CONFIGURATION AND SETUP
####

# Generic config for any NCI user
# specify working directory and storage directory:
wd=$HOME/Projects/PaddockTS
dir=/g/data/$PROJECT/$USER/PaddockTS_Results/ # apologies, it needs the "/" on the end
tmpdir=/scratch/$PROJECT/$USER/tmp3

# # Config for John (old):
# wd=/home/106/jb5097/Projects/PaddockTS
# dir=/g/data/xe2/John/Data/PadSeg/
# tmpdir=/scratch/xe2/jb5097/tmp  

# Config for Chris
# wd=/home/147/cb8590/Projects/PaddockTS
# dir=/g/data/xe2/cb8590/Data/PadSeg/
# tmpdir=/scratch/xe2/cb8590/tmp  

# check/create directories
for d in "$wd" "$dir" "$tmpdir"; do
  if [ -d "$d" ]; then
    echo "Directory already exists: $d "
  else
    echo "Creating directory: $d "
    mkdir -p "$d"
  fi
done

# params to specify Region/Timeframe of interest
# stub=CANAWINDRAb # e.g. <site name>_<buffer>_<years>
# lat=-33.457
# lon=148.679
# buffer=0.01 #this distance in all directions from (lat,lon). 0.01 degrees is ~1km in each direction which woul mean 2kmx2km total
# start='2023-04-01'
# end='2023-08-01'

# params to specify Region/Timeframe of interest
stub=ARBO_taia # e.g. <site name>_<buffer>_<years>
lat=-35.285
lon=149.0662
buffer=0.03 #this distance in all directions from (lat,lon). 0.01 degrees is ~1km in each direction which woul mean 2kmx2km total
start='2024-03-01'
end='2025-03-31'

# params for paddock filtering
min_area_ha=10
max_area_ha=1500
max_perim_area_ratio=40

# Specify which approach for downloading Sentinel-2 data
S2_mode="NCI" # this assumes the user has access to project v10 (and k08?) on NCI, which requires NCI account accessed via ssh
#S2_mode="DEA_ODC" # this is designed for any user that can load virtual environment X [to be created]
#S2_mode="MPC" # this is designed for any user that can load virtual environment X [to be created]

# specify which SAMgeo model to use (see here: https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
samgeo_model='sam_vit_h_4b8939.pth'

# print the config settings:
echo "====================================================================="
echo "Running PaddockTS developmental version with the following settings:"
echo "  Stub:                       $stub"
echo "  Latitude:                   $lat"
echo "  Longitude:                  $lon"
echo "  Buffer (degrees):           $buffer"
echo "  Time range:                 $start to $end"
echo "  Minimum area (ha):          $min_area_ha"
echo "  Maximum area (ha):          $max_area_ha"
echo "  Max perimeter/area:         $max_perim_area_ratio"
echo "  Sentinel-2 download mode:   $S2_mode"
echo "  SAMGeo Model:               $samgeo_model"
echo "  Saving results to:          $dir"
echo "  Saving temporary files to:  $tmpdir"
echo "====================================================================="

# # check if model exists and download it if not, then download it to tmp
# if [ -f "${tmpdir}/${samgeo_model}" ]; then
#   echo "SAMGeo Model found: ${tmpdir}/${samgeo_model}"  
# else
#   echo "SAMGeo Model not found: ${tmpdir}/${samgeo_model}"
#   echo "Downloading model from https://dl.fbaipublicfiles.com/segment_anything/${samgeo_model}..."
#   wget "https://dl.fbaipublicfiles.com/segment_anything/${samgeo_model}" -O "${tmpdir}/${samgeo_model}"
#   if [ $? -eq 0 ]; then
#     echo "Download samgeo complete."
#   else
#     echo "Download samgeo failed."
#     exit 1
#   fi
# fi
# t0=$(date +%s)

# ####
# ## SEQUENTIALLY RUN THE SCRIPTS
# ####

# echo "Step 1: Download Sentinel-2 data from the specified source."
# case "$S2_mode" in

#   "NCI")
#     echo "Running in NCI mode..."
#     echo "Assumes user is on NCI via ssh, and has access to projects v10, k08."
#     module use /g/data/v10/public/modules/modulefiles
#     module load dea/20231204
#     python Code/01_get_Sentinel2_DEA.py \
#       --stub "$stub" \
#       --outdir "$dir" \
#       --lat "$lat" \
#       --lon "$lon" \
#       --buffer "$buffer" \
#       --start_time "$start" \
#       --end_time "$end"
#     module purge
#     ;;

#   "DEA_ODC")
#     echo "Running in DEA_ODC mode..."
#     echo "Hasn't been made yet.. exiting"
#     exit 1
#     # Insert commands/environment setup for DEA_ODC here

#     ;;

#   "MPC")
#     echo "Running in MPC mode..."
#     echo "Hasn't been made yet.. exiting"
#     exit 1
#     # Insert commands/environment setup for MPC here

#     ;;

#   *)
#     echo "Unknown mode: $S2_mode"
#     exit 1
#     ;;
# esac
# echo "Step 1 complete."
# echo
# t1=$(date +%s)
# # Results: 
# # Pickle file representing an xarray object of time series Sentinel2 data downloaded from DEA (<stub>_ds2.pkl)
# # TO DO: should we normalise data so it doesn't matter if it is NBART or just plain L2A?
# # Some metadata about the download (<stub>_ds2_query.pkl)

# echo "Step 2: Download Sentinel-1 data from MPC"
# # The current version will obtain RIO/RIO from the ds2_query.pkl file generated by Code/01_getSentinel2_DEA.py
# # this could be modified to create an RIO directory from the config settings. 
# source /g/data/xe2/John/geospatenv/bin/activate
# python Code/download_S1.py $stub $dir
# deactivate
# echo "Step 2 complete."
# echo
# t2=$(date +%s)

# # Results: 
# # Pickle file representing an xarray object of time series Sentinel1 data downloaded from RIO (<stub>_ds1.pkl)
# # Note: some processed steps required. 
# # Issue: I get random network errors on some runs. Seems that certain scenes are included in the order but then can't be accessed, so it quits. 

# echo "Step 3: calculate indices (and vegetation fractional cover)"
# # This seems to screw up if the modules and python env are not loaded in the right order. Dependency on tensorflow2.15.0 will become an issue for portability
# module load tensorflow/2.15.0 # req for veg frac model
# source /g/data/xe2/John/geospatenv/bin/activate
# python3 Code/02_indices-vegfrac.py --stub $stub --outdir $dir
# module purge
# deactivate
# echo "Step 3 complete."
# echo
# t3=$(date +%s)
# # Results:
# # (<stub>_ds2i.pkl) updated with vegetation indices and vegetation fractional cover.

# echo "Step 4: segment paddocks"
# source /g/data/xe2/John/geospatenv/bin/activate
# python3 Code/03_segment_paddocks.py $stub $dir \
#     --model $tmpdir/$samgeo_model \
#     --min_area_ha $min_area_ha \
#     --max_area_ha $max_area_ha \
#     --max_perim_area_ratio $max_perim_area_ratio
# deactivate
# echo "Complete."
# echo
# t4=$(date +%s)
# # Results:
# # a 3-band image representing Fourier Transform of NDWI time series (<stub>.tif)
# # an image showing thesegmentaiton mask (<stub>_segment.tif)
# # a shapefile of the paddocks (<stub>_segment.gpkg)
# # a shapefile of the paddocks after filtering (<stub>_filt.gpkg) [CHANGE THIS TO <stub>_segment_filt.gpkg]

echo "Step 5: Get environmental variables"
module load gdal/3.6.4 # [check if this is still needed, Chris thought not]
source /g/data/xe2/John/geospatenv/bin/activate
python3 Code/04_environmental.py \
    --stub $stub \
    --outdir $dir \
    --tmpdir $tmpdir \
    --lat $lat \
    --lon $lon \
    --buffer $buffer \
    --start_time $start \
    --end_time $end \
    --nci
module purge
deactivate
echo "Complete."
echo
# Results:
# a tiff file of elevation that can be viewed in QGIS or loaded into python with rasterio (<stub>_terrain.tif)
# a netcdf file of daily rainfall, temperature and evaporation (<stub>_silo_daily.nc)
# a netcdf file of 8 day soil moisture (<stub>_ozwald_8day.nc)

echo "Step 5.5: Generate topographic plots"
# This generates topographic visuals based on the imagery stack, paddock boundaries, and terrain tiff.
module load gdal/3.6.4
source /g/data/xe2/John/geospatenv/bin/activate
python3 Code/topographic_plots.py $stub $dir $tmpdir
module purge
deactivate
echo "Complete."
echo
t5=$(date +%s)
# Results:
# a tiff file & png of: (gaussian smoothed) elevation, topographic index, slope, and aspect

## Checkpoint plots.
module load ffmpeg/4.3.1 # for .mp4 generation
source /g/data/xe2/John/geospatenv/bin/activate
python3 Code/checkpoint_plots.py $stub $dir $tmpdir
t6=$(date +%s)
# Results:
# Set of plots with <stub>_<plot-description>.tif

## 6. Calculate paddock time series
# IN PREP.

## Feature Extraction???
# This is where we will implement functions to estimate things like SoS, EoS, flowering time, from the paddock-level time series data. 

## 7. Generate more outputs
# IN PREP.

####
## PRINT A SUMMARY
####
echo "====================================================================="
echo "Finished running PaddockTS"
total_time=$(( t6 - t_start ))
echo "Total time: ${total_time}s"
echo "Time for each step:"
printf "  Configuration: %ds\n" "$(( t0 - t_start ))"
printf "  Step 1:        %.2f min\n" "$(awk -v sec=$(( t1 - t0 )) 'BEGIN { printf "%.2f", sec/60 }')"
printf "  Step 2:        %.2f min\n" "$(awk -v sec=$(( t2 - t1 )) 'BEGIN { printf "%.2f", sec/60 }')"
printf "  Step 3:        %.2f min\n" "$(awk -v sec=$(( t3 - t2 )) 'BEGIN { printf "%.2f", sec/60 }')"
printf "  Step 4:        %.2f min\n" "$(awk -v sec=$(( t4 - t3 )) 'BEGIN { printf "%.2f", sec/60 }')"
printf "  Step 5:        %.2f min\n" "$(awk -v sec=$(( t5 - t4 )) 'BEGIN { printf "%.2f", sec/60 }')"
printf "  Step 6:        %.2f min\n" "$(awk -v sec=$(( t6 - t5 )) 'BEGIN { printf "%.2f", sec/60 }')"
echo
echo "Output files in order created:"
ls -tr1 "$dir/${stub}"*
echo
echo "Output temporary files in order created:"
ls -tr1 "$tmpdir/${stub}"*
echo "====================================================================="
