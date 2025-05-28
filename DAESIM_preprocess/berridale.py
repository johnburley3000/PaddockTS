import pandas as pd
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
filename = '/g/data/xe2/cb8590/Eucalypts/BERRIDALE_buffer_0.6degrees_min_temp__1889_2025_silo_daily.nc'
ds_silo = xr.load_dataset(filename)
ds_silo.rio.write_crs("EPSG:4326", inplace=True)

# Visualise the number of days below each threshold
save_tif((ds_silo['min_temp'] < -1).sum(dim="time").astype(float), 'silo_days_below_neg1')
save_tif((ds_silo['min_temp'] < -5).sum(dim="time").astype(float), 'silo_days_below_neg5')
save_tif((ds_silo['min_temp'] < -10).sum(dim="time").astype(float), 'silo_days_below_neg10')

# Inspect one of the .nc files
filename = '/g/data/xe2/cb8590/Eucalypts/BERRIDALE_buffer_0.6degrees_Tmin__2000_2024_ozwald_daily_Tmin.nc'
ds_ozwald = xr.load_dataset(filename)
ds_ozwald.rio.write_crs("EPSG:4326", inplace=True)

save_tif((ds_ozwald['Tmin'] < -1).sum(dim="time").astype(float), 'ozwald_days_below_neg1')
save_tif((ds_ozwald['Tmin'] < -5).sum(dim="time").astype(float), 'ozwald_days_below_neg5')
save_tif((ds_ozwald['Tmin'] < -10).sum(dim="time").astype(float), 'ozwald_days_below_neg10')

tmin = ds_ozwald['Tmin']

from matplotlib.animation import FuncAnimation

# +
# %%time
# Generate a gif of the minimum temperature
fig, ax = plt.subplots(figsize=(6, 5))
vmin = float(tmin.min())
vmax = float(tmin.max())
img = ax.imshow(tmin.isel(time=0), cmap='coolwarm', vmin=vmin, vmax=vmax)
cbar = plt.colorbar(img, ax=ax)
title = ax.set_title("")

def update(frame):
    data = tmin.isel(time=frame)
    img.set_array(data)
    time_str = str(tmin.time[frame].values)[:10]
    title.set_text(f"Min Temperature: {time_str}")
    return [img, title]

ani = FuncAnimation(fig, update, frames=len(tmin.time), interval=50, blit=True)

# Save to mp4
filename = '/scratch/xe2/cb8590/tmp/tmin_animation.mp4'
ani.save(filename, fps=20, dpi=150)
plt.close()
# -

import imageio.v3 as iio
from tqdm import tqdm

# +
# %%time

images = []
vmin = float(tmin.min())
vmax = float(tmin.max())

for i in tqdm(range(0, len(tmin.time), 10)):  # sample every 10th frame for faster GIF
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(tmin.isel(time=i), cmap='coolwarm', vmin=vmin, vmax=vmax)
    time_str = str(tmin.time[i].values)[:10]
    ax.set_title(f"Min Temperature: {time_str}")
    plt.axis('off')
    plt.tight_layout()
    fig.canvas.draw()

    # Convert figure to numpy array
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)
    plt.close()

# Save to GIF
filename = '/scratch/xe2/cb8590/tmp/tmin_animation.gif'
iio.imwrite(filename, images, duration=0.1)

# -

limit = 100
ds_small = ds_ozwald.isel(time=slice(0,limit))

# %%time
# Produce time series animation of NDWI:
filename = f'/scratch/xe2/cb8590/tmp/ozwald_xr_animation_{limit}points.mp4'
xr_animation(
    ds=ds_small,
    output_path=filename,
    bands="Tmin",
    interval=100,
    width_pixels=300,
    show_text="Tmin",
)
plt.close()
Video(filename, embed=True)


