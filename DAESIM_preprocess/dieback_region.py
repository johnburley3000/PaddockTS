import pandas as pd
import xarray as xr

# List all the files
# .tsv files contain time series data for each of the 3888 samples
# .nc files contain time series data for a ~50km buffer around Berridale
# .tif files contain static data for a ~50km buffer around Berridale
# !du -sh /g/data/xe2/cb8590/Eucalypts/*

# Inspect one of the tsv files
filename = '/g/data/xe2/cb8590/Eucalypts/EUC_ozwald_daily_Tmax_2020_2024.tsv'
df = pd.read_csv(filename, sep='\t', low_memory=False)
bio_columns = [column for column in df.columns if 'bio' in column]
df = df.drop(columns=bio_columns)
print(f"Number of samples: {df.shape[0]}")
print(f"Number of days: {df.shape[1] - 6}")
df.head()

# Create a geopackage to visualise the tsv results in QGIS


# Inspect one of the .nc files
filename = '/g/data/xe2/cb8590/Eucalypts/BERRIDALE_buffer_0.6degrees_min_temp__1889_2025_silo_daily.nc'
ds = xr.load_dataset(filename)

# +
# Visualise the number of days below negative 1 degrees
days_below_neg1 = (ds['min_temp'] < -1).sum(dim="time")

filename = '/scratch/xe2/cb8590/tmp/silo_below_neg1.tif'
days_below_neg1.rio.to_raster(filename)
print(filename)
