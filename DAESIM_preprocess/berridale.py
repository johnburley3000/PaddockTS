import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


# %%time
from dea_tools.plotting import xr_animation
from IPython.core.display import Video


# List all the files
# .tsv files contain time series data for each of the 3888 samples
# .nc files contain time series data for a ~50km buffer around Berridale
# .tif files contain static data for a ~50km buffer around Berridale
# !du -sh /g/data/xe2/cb8590/Eucalypts/*

# Inspect one of the ozwald tsv files
filename = '/g/data/xe2/cb8590/Eucalypts/EUC_ozwald_daily_Tmin_2020_2024.tsv'
df = pd.read_csv(filename, sep='\t', low_memory=False)
bio_columns = [column for column in df.columns if 'bio' in column]
df = df.drop(columns=bio_columns)
print(f"Number of samples: {df.shape[0]}")
print(f"Number of days: {df.shape[1] - 6}")
df.head()

# Inspect one of the silo tsv files
filename = '/g/data/xe2/cb8590/Eucalypts/EUC_silo_daily_min_temp_2020_2025.tsv'
df = pd.read_csv(filename, sep='\t', low_memory=False)
bio_columns = [column for column in df.columns if 'bio' in column]
df = df.drop(columns=bio_columns)
print(f"Number of samples: {df.shape[0]}")
print(f"Number of days: {df.shape[1] - 6}")
df.head()


def save_tif(ds, name):
    filename = f'/scratch/xe2/cb8590/tmp/{name}.tif'
    ds.rio.to_raster(filename)
    print(filename)


# Inspect one of the .nc files
filename = '/g/data/xe2/cb8590/Eucalypts/BERRIDALE_buffer_0.6degrees_min_temp_1889_2025_silo_daily.nc'
ds_silo = xr.load_dataset(filename)
ds_silo.rio.write_crs("EPSG:4326", inplace=True)

# Visualise the number of days below each threshold
save_tif((ds_silo['min_temp'] < -1).sum(dim="time").astype(float), 'silo_days_below_neg1')
save_tif((ds_silo['min_temp'] < -5).sum(dim="time").astype(float), 'silo_days_below_neg5')
save_tif((ds_silo['min_temp'] < -10).sum(dim="time").astype(float), 'silo_days_below_neg10')

# Inspect one of the .nc files
filename = '/g/data/xe2/cb8590/Eucalypts/BERRIDALE_buffer_0.6degrees_Tmin_2000_2024_ozwald_daily.nc'
ds_ozwald = xr.load_dataset(filename)
ds_ozwald.rio.write_crs("EPSG:4326", inplace=True)

save_tif((ds_ozwald['Tmin'] < -1).sum(dim="time").astype(float), 'ozwald_days_below_neg1')
save_tif((ds_ozwald['Tmin'] < -5).sum(dim="time").astype(float), 'ozwald_days_below_neg5')
save_tif((ds_ozwald['Tmin'] < -10).sum(dim="time").astype(float), 'ozwald_days_below_neg10')

# Load the roads for overlaying with the video
gdf_filename = '/g/data/xe2/cb8590/Open_Street_Maps/berridale_main_roads.gpkg'
gdf = gpd.read_file(gdf_filename)
gdf = gdf[['geometry']]

limit = 365
ds_small = ds_ozwald.isel(time=slice(0,limit))

# %%time
# Count the number of cold days per month
cold_days = ds_ozwald['Tmin'] < -1
cold_days_monthly = cold_days.resample(time='1MS').sum(dim='time')
cold_days_monthly = cold_days_monthly.to_dataset(name='cold_day_count')

# +
# # Create a video of the num cold days per month from ozwald
# filename = '/scratch/xe2/cb8590/tmp/ozwald_number_cold_days_per_month.mp4'
# xr_animation(
#     ds=cold_days_monthly,
#     output_path=filename,
#     bands="cold_day_count",
#     interval=100,
#     width_pixels=300,
#     show_text="OzWALD num days < -1°",
#     show_gdf=gdf[['geometry']],
#     gdf_kwargs={"edgecolor": "black", "linewidth": 1},
#     imshow_kwargs={"cmap": "Blues", "vmin": 0, "vmax": np.max(cold_days_monthly["cold_day_count"].values)},
#     colorbar_kwargs={"colors": "black"}
# )
# plt.close()
# Video(filename, embed=True)
# -

# %%time
# Count the number of cold days per month
cold_days = ds_silo['min_temp'] < -1
cold_years = cold_days.resample(time='1YS').sum(dim='time')
cold_years = cold_years.to_dataset(name='cold_day_count')

# +
# # Create a video of the num cold days per year from SILO
# filename = '/scratch/xe2/cb8590/tmp/silo_number_cold_days_per_year.mp4'
# xr_animation(
#     ds=cold_years,
#     output_path=filename,
#     bands="cold_day_count",
#     interval=200,
#     width_pixels=300,
#     show_text="SILO num days < -1°",
#     show_gdf=gdf[['geometry']],
#     gdf_kwargs={"edgecolor": "black", "linewidth": 1},
#     imshow_kwargs={"cmap": "Blues", "vmin": 0, "vmax": np.max(cold_years["cold_day_count"].values)},
#     colorbar_kwargs={"colors": "black"}
# )
# plt.close()
# Video(filename, embed=True)
# -

import numpy as np

np.max(cold_years['cold_day_count'].values)

# !ls 

# +
# %%time
# Create a tsv of number of frost days between 2020-2022 per lat lon as per Margaret's request

# Load the ozwald data per euc
df1 = pd.read_csv('/g/data/xe2/cb8590/Eucalypts/EUC_ozwald_daily_Tmin_2000_2010.tsv', sep='\t', low_memory=False)
df2 = pd.read_csv('/g/data/xe2/cb8590/Eucalypts/EUC_ozwald_daily_Tmin_2010_2020.tsv', sep='\t', low_memory=False)

# Count the number of days below -1
df1 = df1.drop(columns=bio_columns + ["species", "datum", "elevation"])
date_cols = df1.columns[3:]
df1['frost_days'] = (df1[date_cols] < -1).sum(axis=1)
df1_frost = df1.drop(columns = date_cols)

df2 = df2.drop(columns=bio_columns + ["species", "datum", "elevation"])
date_cols = df2.columns[3:]
df2['frost_days'] = (df2[date_cols] < -1).sum(axis=1)
df2_frost = df2.drop(columns = date_cols)
# -

# Sum the frost days from 2000-2010 and 2010-2020
df_frost = df1_frost.merge(df2_frost.drop(columns=['X', 'Y']), on='sample_name', how='left')
df_frost['frost_days'] = df_frost['frost_days_x'] +  df_frost['frost_days_y']
df_frost = df_frost.drop(columns = ['frost_days_x', 'frost_days_y'])

# Save as a tsv file
filename = '/scratch/xe2/cb8590/Eucalypts/EUC_frost_days_ozwald_2000-2020.tsv'
df_frost.to_csv(filename, sep='\t', index=False)
print(filename)

# Save as a geopackage
gdf = gpd.GeoDataFrame(
    df_frost,
    geometry=gpd.points_from_xy(df_frost.X, df_frost.Y),
    crs="EPSG:4326"
)
filename = '/scratch/xe2/cb8590/Eucalypts/EUC_frost_days_ozwald_2000-2020.gpkg'
gdf.to_file(filename, layer='frost_days', driver="GPKG")
filename

# Create a dataframe of frost days for that region Margaret sent
cold_days = ds_ozwald['Tmin'] < -1
ds_frost = cold_days.sum(dim='time')

df_frost_region = ds_frost.to_dataframe(name='frost_days').reset_index()
df_frost_region = df_frost_region.drop(columns = ['spatial_ref'])
df_frost_region = df_frost_region.rename(columns={'longitude': 'X', 'latitude': 'Y'})
df_frost_region = df_frost_region[['X', 'Y', 'frost_days']]

filename = '/scratch/xe2/cb8590/Eucalypts/BERRIDALE_frost_days_ozwald_2000-2020.tsv'
df_frost_region.to_csv(filename, sep='\t')
print(filename)

# Save as a geopackage
gdf = gpd.GeoDataFrame(
    df_frost_region,
    geometry=gpd.points_from_xy(df_frost_region.X, df_frost_region.Y),
    crs="EPSG:4326"
)
filename = '/scratch/xe2/cb8590/Eucalypts/BERRIDALE_frost_days_ozwald_2000-2020.gpkg'
gdf.to_file(filename, layer='frost_days', driver="GPKG")
filename
