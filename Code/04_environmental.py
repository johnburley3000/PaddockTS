"""
Description:
This script downloads elevation, soils, climate and canopy heights

Requirements:
- Modules: gdal/3.6.4  (for terrain_tiles gdalwarp)
- Environment base: /g/data/xe2/John/geospatenv

Additionally, for the canopy_height.download_new_tiles() to work you need to create an AWS account, then create a file named .aws/credentials in your home directory (e.g. /home/147/cb8590) with these contents:
[default]
aws_access_key_id = ACCESS_KEY
aws_secret_access_key = SECRET_KEY

"""
import argparse
import logging
import os
import sys

# Change directory and insert it into the python path
paddockTS_dir = os.path.join(os.path.expanduser('~'), "Projects/PaddockTS")
os.chdir(paddockTS_dir)
sys.path.append(paddockTS_dir)

from DAESIM_preprocess.terrain_tiles import terrain_tiles
from DAESIM_preprocess.slga_soils import slga_soils, asris_urls
from DAESIM_preprocess.ozwald_yearly import ozwald_yearly_average
from DAESIM_preprocess.ozwald_8day import ozwald_8day, ozwald_8day_abbreviations
from DAESIM_preprocess.ozwald_daily import ozwald_daily, ozwald_daily_abbreviations
from DAESIM_preprocess.silo_daily import silo_daily
from DAESIM_preprocess.canopy_height import canopy_height


# Adjust logging configuration for the script
logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Download environmental variables for the region of interest and save as lots of .tif and .nc files
        
Example usage:
python3 Code/04_environmental.py --stub test --outdir /g/data/xe2/cb8590 --tmpdir /scratch/xe2/cb8590 --lat -34.3890 --lon 148.4695 --buffer 0.01 --start_time '2020-01-01' --end_time '2020-03-31'""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--stub", type=str, required=True, help="Stub name for file naming")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for saved files")
    parser.add_argument("--tmpdir", type=str, required=True, help="Directory for temporary intermediate files")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the center of the area of interest")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the center of the area of interest")
    parser.add_argument("--buffer", type=float, required=True, help="Buffer in degrees to define the area around the center point")
    parser.add_argument("--start_time", type=str, required=True, help="Start time for the data query (YYYY-MM-DD)")
    parser.add_argument("--end_time", type=str, required=True, help="End time for the data query (YYYY-MM-DD)")
    return parser.parse_args()

        
def main(args):
    lat = args.lat
    lon = args.lon
    buffer = args.buffer
    stub = args.stub
    start_year = args.start_time[:4]
    end_year = args.end_time[:4]
    outdir = args.outdir
    tmpdir = args.tmpdir

    terrain_tiles(lat, lon, buffer, outdir, stub, tmpdir)

    variables = ['Clay', 'Sand', 'Silt', 'pH_CaCl2']
    slga_soils(variables, lat, lon, buffer, outdir, stub)

    # NCI Thredds not working via PBS for some reason. Have to rewrite these two functions to access the directories directly like with ozwald_8day below.
    # ozwald_yearly_average(["Tmax", "Tmin", "Pg"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir)
    # ozwald_daily(["VPeff", "Uavg"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir)

    variables = ['Ssoil']
    ozwald_8day(variables, lat, lon, buffer, start_year, end_year, outdir, stub)

    variables = ["daily_rain", "max_temp", "min_temp", "et_morton_actual", "et_morton_potential"]
    ds_silo_daily = silo_daily(variables, lat, lon, buffer, start_year, end_year, outdir, stub)

    canopy_height(lat, lon, buffer, outdir, stub, tmpdir)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
