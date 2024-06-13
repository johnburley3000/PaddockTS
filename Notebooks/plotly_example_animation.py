import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime

# +
# Load the xarray
silo_filepath = "/g/data/ub8/au/SILO/daily_rain/2023.daily_rain.nc"
silo_full = xr.open_dataset(silo_filepath)

# Region of interest
lat = -34.38904277303204
lon = 148.46949938279096
buffer = 1

silo_region = silo_full.sel(lat=slice(lat - buffer, lat + buffer), lon=slice(lon - buffer, lon + buffer))
ds = silo_region.sel(time=slice('2023-01-01', '2023-12-01'))

rainfall = ds['daily_rain'] 
times = ds['time']

# +
# Create the animation
fig, ax = plt.subplots(figsize=(10, 6))
def animate(time_index):
    ax.clear()
    time_slice = rainfall.isel(time=time_index)
    c = ax.pcolormesh(time_slice['lon'], time_slice['lat'], time_slice, shading='auto', cmap='Blues')
    if not hasattr(ax, 'colorbar'):
        ax.colorbar = fig.colorbar(c, ax=ax, label='Rainfall')
    ax.set_title(f"Rainfall Data on {str(times[time_index].values)[:10]}")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

start = datetime.now()
ani = animation.FuncAnimation(fig, animate, frames=len(times), interval=33)
ani.save('rainfall_animation.mp4', writer='ffmpeg')
end = datetime.now()

print(end - start)
# -


