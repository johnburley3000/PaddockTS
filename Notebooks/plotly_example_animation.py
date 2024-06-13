# +
import numpy as np
import pandas as pd
import plotly.express as px

# Example data
times = pd.date_range('2023-01-01', periods=10, freq='D')
latitudes = np.linspace(-90, 90, 20)
longitudes = np.linspace(-180, 180, 20)
rainfall_data = np.random.rand(len(times), len(latitudes), len(longitudes))

data = []
for t, time in enumerate(times):
    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            data.append([time, lat, lon, rainfall_data[t, i, j]])

df = pd.DataFrame(data, columns=['time', 'lat', 'lon', 'rainfall'])

# Create the plotly figure
fig = px.scatter_geo(df, 
                     lon='lon', 
                     lat='lat', 
                     color='rainfall', 
                     animation_frame='time',
                     color_continuous_scale='Blues',
                     projection='natural earth',
                     title='Time Series Rainfall Data')

fig.update_layout(
    geo=dict(
        showland=True,
        landcolor="white",
        showocean=True,
        oceancolor="lightblue"
    ),
    coloraxis_colorbar=dict(
        title="Rainfall"
    )
)

# Save the animation as an HTML file
fig.write_html("rainfall_animation.html")
