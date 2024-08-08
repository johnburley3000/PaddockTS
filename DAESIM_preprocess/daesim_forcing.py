# This script gathers data from OzWald and ANUClimate for a single location over the specified years.

import os
import pandas as pd

print(os.getcwd())

# Need to change directory to whichever folder contains the python files for the imports to work
os.chdir('/home/147/cb8590/Projects/PaddockTS/Notebooks/DAESIM_preprocess')

from ozwald_8day import ozwald_8day_multivariable
from ozwald_daily import ozwald_daily_multivariable
from anuclimate_daily import anuclimate_multivariable

latitude=-34.3890427
longitude=148.469499
years=["2020","2021"]

# %%time
df_ozwald_daily = ozwald_daily_multivariable(["VPeff", "Uavg"], latitude, longitude, years)
df_ozwald_daily.head()

# %%time
df_ozwald_8day = ozwald_8day_multivariable(["Ssoil", "Qtot", "LAI", "GPP"], latitude, longitude, years)
df_ozwald_8day.head()

# +
# %%time

# anuclimate takes longer because files are stored monthly instead of yearly
df_anuclimate = anuclimate_multivariable(["rain", "tmin", "tmax", "srad"], latitude, longitude, years)
df_anuclimate.head()
# -

df_daesim_forcing = pd.concat([df_ozwald_daily, df_ozwald_8day, df_anuclimate], axis=1)
df_daesim_forcing.head()

df_daesim_forcing.to_csv("DAESim_forcing.csv")
