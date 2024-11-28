# +
import pandas as pd
import psutil
import os
import numpy as np
from datetime import datetime, timedelta

process = psutil.Process(os.getpid())
memory_in_mb = process.memory_info().rss / 1024 ** 2  # Convert bytes to MB
print(f"Memory usage: {memory_in_mb:.2f} MB")

base_date = datetime(1970, 1, 1)

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

interesting_columns = [ 'METADom_GERMINATION.RAIN.DATE',
 'METADom_Year',
 'METADom_Sow_Year',
 'METADom_Crop_rotation_minus_0',
 'METADom_Crop_rotation_minus_1',
 'METADom_Crop_rotation_minus_2',
 'METADom_Crop_rotation_minus_3',
 'METADom_Crop_rotation_minus_4',
 'METADom_Crop_rotation_minus_5',
 'METADom_Crop_rotation_minus_6',
 'METADom_metadata_exists.x',
 'PHENDom_MEAN.YIELD',
 'PHENDom_LSD',
 'PHENDom_CV',
 'PHENDom_REPORT.DATE',
 'PHENDom_PROBABILITY',
 'PHENDom_Harvest_date_numeric',
 'METADom_metadata_exists.y',
 'Trial_date_Index',
 'PHENDom_Days_to_Harvest',
 'PHENDom_First_Flower',
 'PHENDom_Height_cm',
 'PHENDom_Height_lowest_pod',
 'PHENDom_Hundred_Seed_Weight',
 'PHENDom_Predicted_Yield',
 'PHENDom_Predicted_Yield_MET',
 'PHENDom_Predicted_Yield_Single_Site',
 'PHENDom_Rows_per_Plot',
 'PHENDom_Thousand_grain_weight',
 'PHENDom_Thousand_grain_weight_obs_2',
 'PHENDom_Tillering_Score',
 'PHENDom_Trial.Comments',
 'PHENDom_Trial.Comments_1',
 'PHENDom_TrialCode',
 'PHENDom_V12',
 'PHENDom_Trial_comments_2',
 'PHENDom_Trial_comments_3',
 'PHENDom_Trial_comments_4',
 'PHENDom_Victoria_Blight',
 'PHENDom_Vigour',
 'PHENDom_Waterlogging',
 'PHENDom_Weed_score',
 'PHENDom_Weed_score_obs_2',
 'PHENDom_X100.Seed.Weight_2',
 'PHENDom_X1000.grain.weight_2',
 'PHENDom_X1000.grain.weight_3',
 'PHENDom_X1000.grain.weight_6',
 'PHENDom_Yellow_Leaf_Spot',
 'PHENDom_yield_pct_of_average',
 'PHENDom_yield_t_ha',
 'PHENDom_Zadoks_score',
 'ENVDom_Grazing_damage',
 'PHENDom_Crop',
 'METADom_previous_crop_same',
 'PHENDom_Pins_Crop',
 'PHENDom_Pins_Site_Mean_Yield'
            ]

# +
most_interesting_columns = [
 'BOMDom_Trial_ID',
 'BOMDom_Sowing_date_numeric',
 'BOMDom_Latitude',
 'BOMDom_Longitude',
 'PHENDom_MEAN.YIELD',
 'PHENDom_Days_to_Harvest',
 'PHENDom_Crop',
            ]

date_threshold = datetime(2016, 1, 1)


# +
# df = df_canola[most_interesting_columns]
# df.drop_duplicates(inplace=True)
# df = df.dropna(axis=0, how='all')
# df['Sowing_Date'] = df['BOMDom_Sowing_date_numeric'].apply(lambda x: base_date + timedelta(days=x))
# df = df[(df['PHENDom_Crop'] == 'Canola') & (df['Sowing_Date'] > date_threshold) & df['PHENDom_MEAN.YIELD'].notna()]
# -

df_site_frame['PHENDom_Crop'].unique()

# +
df = df_site_frame[most_interesting_columns]
df['Sowing_Date'] = df['BOMDom_Sowing_date_numeric'].apply(lambda x: base_date + timedelta(days=x))
df = df[(df['PHENDom_Crop'] == 'Canola') & (df['Sowing_Date'] > date_threshold) & df['PHENDom_MEAN.YIELD'].notna()]

filename = os.path.join(scratch_dir, "Canola_2016-2018.csv")
df.to_csv(filename, index=False)
print(filename)

# +
df = df_site_frame[most_interesting_columns]
df['Sowing_Date'] = df['BOMDom_Sowing_date_numeric'].apply(lambda x: base_date + timedelta(days=x))
df = df[(df['PHENDom_Crop'] == 'Wheat') & (df['Sowing_Date'] > date_threshold) & df['PHENDom_Days_to_Harvest'].notna()]

filename = os.path.join(scratch_dir, "Wheat_2016-2018.csv")
df.to_csv(filename, index=False)
print(filename)

# +
valid_counts = df_canola.replace(np.nan).count(axis=1)
len(df_canola.columns), valid_counts.max()

df = df_canola.loc[valid_counts.argmax()]
df = pd.DataFrame([df])

filename = os.path.join(scratch_dir, "single_row_all_columns.csv")
df.to_csv(filename, index=False)
print(filename)
# -

print(os.path.join(scratch_dir, "Canola_2016-2018.csv"))
print(os.path.join(scratch_dir, "Wheat_2016-2018.csv"))
print(os.path.join(scratch_dir, "single_row_all_columns.csv"))


