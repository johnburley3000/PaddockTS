# +
import pandas as pd
import psutil
import os
import numpy as np
from datetime import datetime, timedelta

process = psutil.Process(os.getpid())
memory_in_mb = process.memory_info().rss / 1024 ** 2  # Convert bytes to MB
print(f"Memory usage: {memory_in_mb:.2f} MB")
# -

pd.set_option('display.max_columns', 100)


scratch_dir = '/scratch/xe2/cb8590/'

# !ls /g/data/xe2/cb8590/NVT

filename = "/g/data/xe2/cb8590/NVT/site_frame.csv"
df_site_frame = pd.read_csv(filename, encoding='latin1', low_memory=False)
df_site_frame

filename = "/g/data/xe2/cb8590/NVT/Wheat_part1.csv"
df_wheat1 = pd.read_csv(filename, encoding='latin1', low_memory=False)
df_wheat1

filename = "/g/data/xe2/cb8590/NVT/Canola.csv"
df_canola = pd.read_csv(filename, encoding='latin1', low_memory=False)
df_canola

most_interesting_columns = [
 'METADom_Sow_Year',
 'BOMDom_Trial_ID',
 'BOMDom_Sowing_date_numeric',
 'BOMDom_Latitude',
 'BOMDom_Longitude',
 'PHENDom_MEAN.YIELD',
 'PHENDom_Days_to_Harvest',
 'PHENDom_Crop',
            ]

df = df_canola[most_interesting_columns]
df = df[df['METADom_Sow_Year'] >= 2016]
df.drop_duplicates(inplace=True)

df[most_interesting_columns].describe()

df.dropna(axis=1, how='all').dropna(axis=0, how='all')


