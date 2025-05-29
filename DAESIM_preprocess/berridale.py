import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt


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

# Create a video of the num cold days per month from ozwald
filename = '/scratch/xe2/cb8590/tmp/ozwald_number_cold_days_per_month.mp4'
xr_animation(
    ds=cold_days_monthly,
    output_path=filename,
    bands="cold_day_count",
    interval=100,
    width_pixels=300,
    show_text="OzWALD num days < -1°",
    show_gdf=gdf[['geometry']],
    gdf_kwargs={"edgecolor": "black", "linewidth": 1},
    imshow_kwargs={"cmap": "Blues"},
    colorbar_kwargs={"colors": "black"}
)
plt.close()
Video(filename, embed=True)

# %%time
# Count the number of cold days per month
cold_days = ds_silo['min_temp'] < -1
cold_years = cold_days.resample(time='1YS').sum(dim='time')
cold_years = cold_years.to_dataset(name='cold_day_count')

# Create a video of the num cold days per year from SILO
filename = '/scratch/xe2/cb8590/tmp/silo_number_cold_days_per_year.mp4'
xr_animation(
    ds=cold_years,
    output_path=filename,
    bands="cold_day_count",
    interval=200,
    width_pixels=300,
    show_text="SILO num days < -1°",
    show_gdf=gdf[['geometry']],
    gdf_kwargs={"edgecolor": "black", "linewidth": 1},
    imshow_kwargs={"cmap": "Blues"},
    colorbar_kwargs={"colors": "black"}
)
plt.close()
Video(filename, embed=True)

# Inspect one of the .nc files
filename = '/g/data/xe2/cb8590/Eucalypts/BERRIDALE_buffer_0.6degrees_Tmax_2000_2024_ozwald_daily.nc'
ds_ozwald_max = xr.load_dataset(filename)
ds_ozwald_max.rio.write_crs("EPSG:4326", inplace=True)

hot_days = ds_ozwald_max['Tmax'] > 30
hot_months = hot_days.resample(time='1MS').sum(dim='time')
hot_months = hot_months.to_dataset(name='hot_day_count')

filename = '/scratch/xe2/cb8590/tmp/ozwald_number_hot_days_per_month.mp4'
xr_animation(
    ds=hot_months,
    output_path=filename,
    bands="hot_day_count",
    interval=100,
    width_pixels=300,
    show_text="OzWALD num days > 30°",
    show_gdf=gdf[['geometry']],
    gdf_kwargs={"edgecolor": "black", "linewidth": 1},
    imshow_kwargs={"cmap": "Reds"},
    colorbar_kwargs={"colors": "black"}
)
plt.close()
Video(filename, embed=True)

# SILO hot days per year
filename = '/g/data/xe2/cb8590/Eucalypts/BERRIDALE_buffer_0.6degrees_max_temp_1889_2025_silo_daily.nc'
ds_silo_max = xr.load_dataset(filename)
ds_silo_max.rio.write_crs("EPSG:4326", inplace=True)

hot_days_silo = ds_silo_max['max_temp'] > 30
hot_years_silo = hot_days_silo.resample(time='1YS').sum(dim='time')
hot_years_silo = hot_years_silo.to_dataset(name='hot_day_count')

filename = '/scratch/xe2/cb8590/tmp/silo_number_hot_days_per_year.mp4'
xr_animation(
    ds=hot_years_silo,
    output_path=filename,
    bands="hot_day_count",
    interval=200,
    width_pixels=300,
    show_text="SILO num days > 30°",
    show_gdf=gdf[['geometry']],
    gdf_kwargs={"edgecolor": "black", "linewidth": 1},
    imshow_kwargs={"cmap": "Reds"},
    colorbar_kwargs={"colors": "black"}
)
plt.close()
Video(filename, embed=True)


