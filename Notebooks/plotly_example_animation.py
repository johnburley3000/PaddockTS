# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example data: create a DataFrame with random rainfall data
times = pd.date_range('2023-01-01', periods=100, freq='D')
latitudes = np.linspace(-90, 90, 100)
longitudes = np.linspace(-180, 180, 100)

rainfall_data = np.random.rand(100, 100, 100)  # [time, lat, lon]

# Convert to a DataFrame for easier manipulation (optional)
df = pd.DataFrame({
    'time': np.repeat(times, 100*100),
    'lat': np.tile(np.repeat(latitudes, 100), 100),
    'lon': np.tile(longitudes, 100*100),
    'rainfall': rainfall_data.flatten()
})

# -

def plot_rainfall(time_index):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_slice = df[df['time'] == times[time_index]]
    pivot_table = time_slice.pivot(index='lat', columns='lon', values='rainfall')
    
    c = ax.pcolormesh(pivot_table.columns, pivot_table.index, pivot_table.values, shading='auto', cmap='Blues')
    fig.colorbar(c, ax=ax, label='Rainfall')
    ax.set_title(f"Rainfall Data on {times[time_index].strftime('%Y-%m-%d')}")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.close(fig)  # Close the figure to avoid display in the notebook
    return fig



# +
from PIL import Image

frames = []
for t in range(len(times)):
    fig = plot_rainfall(t)
    fig.canvas.draw()
    
    # Convert to a PIL image and append to frames
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(Image.fromarray(image))

# Save frames as a GIF
frames[0].save('rainfall_animation.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)


# +
from IPython.display import Image as IPImage
from IPython.display import display

display(IPImage(filename='rainfall_animation.gif'))


# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Example data
times = pd.date_range('2023-01-01', periods=10, freq='D')
latitudes = np.linspace(-90, 90, 20)
longitudes = np.linspace(-180, 180, 20)

# Create a 3D array of random rainfall data (time, lat, lon)
rainfall_data = np.random.rand(len(times), len(latitudes), len(longitudes))

# Flatten the data and create a DataFrame
data = []
for t, time in enumerate(times):
    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            data.append([time, lat, lon, rainfall_data[t, i, j]])

df = pd.DataFrame(data, columns=['time', 'lat', 'lon', 'rainfall'])

# Define the plotting function
def plot_rainfall(ax, time_index):
    ax.clear()
    time_slice = df[df['time'] == times[time_index]]
    pivot_table = time_slice.pivot(index='lat', columns='lon', values='rainfall')
    
    c = ax.pcolormesh(pivot_table.columns, pivot_table.index, pivot_table.values, shading='auto', cmap='Blues')
    if not hasattr(ax, 'colorbar'):
        ax.colorbar = fig.colorbar(c, ax=ax, label='Rainfall')
    ax.set_title(f"Rainfall Data on {times[time_index].strftime('%Y-%m-%d')}")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    return ax



# +
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define the animation function
def animate(t):
    plot_rainfall(ax, t)

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=len(times), interval=300)

# Save the animation as an MP4 file
ani.save('rainfall_animation.mp4', writer='ffmpeg')

# -


