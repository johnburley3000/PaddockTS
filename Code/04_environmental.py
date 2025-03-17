"""
Description:
This script downloads elevation, soils, climate and canopy heights

Requirements:
- Modules: gdal/3.6.4  (for terrain_tiles gdalwarp)
- Environment base: /g/data/xe2/John/geospatenv

"""
import argparse
import logging
import os
import sys

# Change directory to the PaddockTS repo
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    paddockTS_dir = os.path.join(os.path.expanduser("~"), "Projects/PaddockTS")
elif os.path.basename(os.getcwd()) != "PaddockTS":
    paddockTS_dir = os.path.dirname(os.getcwd())  # Running in a jupyter notebook 
else:  # Already running locally from PaddockTS root
    paddockTS_dir = os.getcwd()
os.chdir(paddockTS_dir)
sys.path.append(paddockTS_dir)

from DAESIM_preprocess.terrain_tiles import terrain_tiles
from DAESIM_preprocess.slga_soils import slga_soils, slga_soils_abbrevations
from DAESIM_preprocess.ozwald_8day import ozwald_8day, ozwald_8day_abbreviations
from DAESIM_preprocess.ozwald_daily import ozwald_daily, ozwald_daily_abbreviations
from DAESIM_preprocess.silo_daily import silo_daily, silo_abbreviations
from DAESIM_preprocess.daesim_forcing import daesim_forcing, daesim_soils

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

    thredds=True

    ozwald_daily(["Uavg", "VPeff"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)
    ozwald_daily(["Tmax", "Tmin"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)
    ozwald_daily(["Pg"], lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)

    variables = ["Ssoil", "Qtot", "LAI", "GPP"]
    ozwald_8day(variables, lat, lon, buffer, start_year, end_year, outdir, stub, tmpdir, thredds)

    # variables = ["radiation", "vp", "max_temp", "min_temp", "daily_rain", "et_morton_actual", "et_morton_potential"]
    variables = ["radiation"]
    ds_silo_daily = silo_daily(variables, lat, lon, buffer, start_year, end_year, outdir, stub)

    df_climate = daesim_forcing(outdir, stub)

    variables = ['Clay', 'Sand', 'Silt', 'pH_CaCl2']
    variables = ['Clay', 'Silt', 'Sand', 'pH_CaCl2', 'Bulk_Density', 'Available_Water_Capacity', 'Effective_Cation_Exchange_Capacity', 'Total_Nitrogen', 'Total_Phosphorus']
    depths=['5-15cm', '15-30cm', '30-60cm', '60-100cm']
    slga_soils(variables, lat, lon, buffer, outdir, stub, depths)

    df_soils = daesim_soils(outdir, stub)

    terrain_tiles(lat, lon, buffer, outdir, stub, tmpdir)

if __name__ == "__main__":
    args = parse_arguments()
    # args = argparse.Namespace(
    #     lat=-34.3890427,
    #     lon=148.469499,
    #     buffer=0.1,
    #     stub="Test",
    #     start_time="2020-01-01",
    #     end_time="2021-12-31",
    #     outdir=".",
    #     tmpdir="."
    # )
    main(args)


