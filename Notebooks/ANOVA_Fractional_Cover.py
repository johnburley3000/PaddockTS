

# +
import os
import shutil
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import scipy.stats as stats
# -

pd.set_option('display.max_rows', None)

os.chdir('./Projects/PaddockTS')

# +
stub = 'MILG_b033_2017-24'
chris_outdir = "/g/data/xe2/cb8590/Data/PadSeg/"
john_outdir =  "/g/data/xe2/John/Data/PadSeg/"  # Chris can read, but can't write to John's directory
tmp_dir = '/scratch/xe2/cb8590/tmp/'+stub+'/'

paddocks_manual = "/g/data/xe2/John/Data/PadSeg/milg_manualpaddocks2.gpkg" # hand-drawn paddock polygons with name column that MAY match with annotation data (not all rows will have annotations)
paddock_annotations = "/g/data/xe2/John/Data/PadSeg/MILG_paddocks_tmp.csv" # the latest version of paddock management annotation data (assumes format stays the same!)

# -

# Check memory usage
import resource
def memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return f"Memory usage: {usage.ru_maxrss / 1024} MB"
print(memory_usage())

# +
# # Read in the polygons from SAMGeo (these will not neccesarily match user-provided paddocks)
# pol = gpd.read_file(outdir+stub+'_filt.gpkg')
# pol['paddock'] = range(1,len(pol)+1)
# pol['paddock'] = pol.paddock.astype('category')

# +
# Read in manual polygons and paddock annotation data. Merge and keep as a geopandas df:

# paddock annotatioun data:
pad_an = pd.read_csv(paddock_annotations)

# Load the manual drawn polygons GeoDataFrame
pad_man = gpd.read_file(paddocks_manual)

# Remove rows that have no geometry
pad_man = pad_man[pad_man.geometry.notnull()]

# Remove rows with invalid geometries
pad_man = pad_man[pad_man.is_valid]

# Add a new column with unique numbers
pad_man['paddock'] = range(1, len(pad_man) + 1)

# merge manual drawn polygons with annotations
pad_manan = pd.merge(pad_man, pad_an, left_on='name', right_on='Name', how='left').drop(columns=['Name'])
pad_manan.iloc[:1]
# -

pvt = np.load(chris_outdir+stub+'_pvt_manual.npy')
pvt.shape

# %%time
with open(john_outdir+stub+'_ds2.pkl', 'rb') as handle:
    ds_full = pickle.load(handle)
memory_usage()

with open(chris_outdir+stub+'_ds_weekly_paddocks_'+'all_years'+'.pkl', 'rb') as handle:
    ds_weekly_paddocks = pickle.load(handle)
ds_weekly_paddocks

# +
## Cleaning the paddocks dataframe

# named_paddocks = pad_manan[pad_manan['type'] == 'named']
named_paddocks = pad_manan   # Retrospectively decided to apply this cleaning to the whole table, so a bad variable name

# Remove columns we don't need
named_dropped = named_paddocks.drop(columns=["geometry", '2020_Impr. Graz Man', '2018_Cover Crops', "paddock", '2019_Pasture Improv', '2019_Cover Crops', '2021_Pasture Improv', '2019_Impr. Graz Man',  'Arable Area (ha)', 'Soil Tested', 'CMB applied', '2025_Crop', '2017_Cover Crops', '2017_Pasture Improv', '2018_Impr. Graz Man', '2017_Impr. Graz Man', '2025_Pasture Improv', '2025_Impr. Graz Man', '2025_Cover Crops', '2024_Crop', '2024_Pasture Improv', '2024_Impr. Graz Man',
       '2024_Cover Crops', '2023_Impr. Graz Man'])
named_dropped.rename(columns={' 2021 Crop': '2021_Crop'}, inplace=True)
named_dropped = named_dropped.fillna('')

# Cleanup 'x' values with the column name
years = ['2017', '2018', '2019', '2020', '2021', '2022', '2023']
for year in years:
    column = f'{year}_Pasture Improv'
    if column in named_dropped:
        named_dropped[column] = named_dropped[column].replace('x', 'Improved Pasture')
        
    column = f'{year}_Impr. Graz Man'
    if column in named_dropped:
        named_dropped[column] = named_dropped[column].replace('x', 'Managed Grazing')

    column = f'{year}_Cover Crops'
    if column in named_dropped:
        named_dropped[column] = named_dropped[column].replace('x', 'Cover Crop')

# There are two columns for 2021. Rename one of them to make them unique
unduplicated_columns = list(named_dropped.columns)
unduplicated_columns[2] = '2021_Crop_2'
named_dropped.columns = unduplicated_columns

# Remove whitespaces
named_dropped = named_dropped.astype(str)
for column in named_dropped:
    named_dropped[column].apply(lambda x: x.strip() if isinstance(x, str) else x)

# Combine management actions into a single column per year
for year in years:
    # suffixes = '_Crop', '_Crop_2',  '_Pasture Improv', '_Impr. Graz Man', '_Cover Crops'
    suffixes = '_Crop', '_Crop_2', '_Cover Crops'
    columns = []
    for suffix in suffixes:
        column = f'{year}{suffix}'
        if column in named_dropped:
            columns.append(column)
    named_dropped[year] = named_dropped[columns].apply(lambda x: ', '.join(x).strip().strip(",").strip().strip(",").strip().strip(","), axis=1)

# Remove unwanted columns
named_dropped.rename(columns={'name': 'Name'}, inplace=True)
final_columns = ['2017', '2018', '2019',
       '2020', '2021', '2022', '2023', 'Pasture State', 'type', 'Name']
df = named_dropped[final_columns]

# Rename the cropping paddocks to tell them apart
df['Name'][df['Pasture State'] == 'Cropping'] = "*Cropping - " + df['Name']
df['type'][df['Pasture State'] == 'Cropping'] = 'cropping'

df = df.sort_values(['Name'])

# +
# # Create some row names for consistency across years
# ds_weekly_paddocks_named = ds_weekly_paddocks.where(ds_weekly_paddocks['type'] == 'named', drop=True)
# plot_clustermap_frac(ds_paddocks=ds_weekly_paddocks_named, variable_names=['bare', 'photosynthetic', 'non_photosynthetic'], exaggerate=True)
# ds_paddocks=ds_weekly_paddocks_named
ds_paddocks=ds_weekly_paddocks
variable_names=['bare', 'photosynthetic', 'non_photosynthetic']
exaggerate=True


pt_variables = []
for variable in variable_names:
    pt_variable = ds_paddocks.sel(variable=variable).pvt.values
    pt_variable = np.apply_along_axis(
        lambda x: pd.Series(x).interpolate(method='linear', limit_direction='both').to_numpy(), 
        axis=1, 
        arr=pt_variable
    )
    pt_variable = np.nan_to_num(pt_variable, nan=0)
    pt_variables.append(pt_variable)

pt_bare = pt_variables[0]
pt_photosynthetic = pt_variables[1]
pt_non_photosynthetic = pt_variables[2]

if exaggerate:
    # This exaggerate any variables that don't show up much
    pt_bare_norm = (pt_bare - pt_bare.min()) / (pt_bare.max() - pt_bare.min())
    pt_photosynthetic_norm = (pt_photosynthetic - pt_photosynthetic.min()) / (pt_photosynthetic.max() - pt_photosynthetic.min())
    pt_non_photosynthetic_norm = (pt_non_photosynthetic - pt_non_photosynthetic.min()) / (pt_non_photosynthetic.max() - pt_non_photosynthetic.min())
    rgb_image = np.dstack((pt_bare_norm, pt_photosynthetic_norm, pt_non_photosynthetic_norm))
else:
    rgb_image = np.dstack((pt_bare, pt_photosynthetic, pt_non_photosynthetic))
        
# Extract the timestamps and convert to Pandas DatetimeIndex
time_stamps = ds_paddocks.time.values
time_index = pd.to_datetime(time_stamps)

# Adjust start date to include January if necessary
start_date = time_index.min()
if start_date.month != 1:
    start_date = pd.Timestamp(year=start_date.year, month=1, day=1)

# Ensure January is included in monthly_start
monthly_start = pd.date_range(start=start_date, end=time_index.max(), freq='MS')

# Find the closest previous timestamps in the original time_index
monthly_ticks = []
for date in monthly_start:
    prior_dates = time_index[time_index <= date]
    if not prior_dates.empty:
        monthly_ticks.append(prior_dates[-1])

monthly_ticks_str = [str(t)[:10] for t in monthly_ticks]

# Extract the crop type information for the specified year
# crop_col = f'{year}_Crop'
# the_crops = pad_manan[pad_manan['type'] == 'named'][crop_col].fillna('')
# the_crops = the_crops.apply(lambda x: x.strip() if isinstance(x, str) else x).replace('', '')

# Get paddock names
row_names = ds_paddocks.name.values

names_crops = row_names  # We could add more info here about each paddock
# df = df_sorted


# # # Order the rows by clustering
# Z = linkage(pt_photosynthetic, method='average', metric='euclidean')
# row_order = leaves_list(Z)
# ordered_row_names = list(df['Name'][row_order])

# # # Order the rows alphabetically
row_order = df.index.to_list()
ordered_row_names = list(df['Name'])

# Reorder the RGB image
rgb_image_ordered = rgb_image[row_order, :, :]
print(row_order)
    
# Calculate the aspect ratio
num_rows, num_cols = rgb_image_ordered.shape[:2]
aspect_ratio = ((num_cols / num_rows)/ 5) * 89/25
# aspect_ratio = ((num_cols / num_rows)/ 5)


# Plot the RGB heatmap
fig, ax = plt.subplots(figsize=(50, 50))
im = ax.imshow(rgb_image_ordered, aspect=aspect_ratio)

# Customize the plot
font_size = 12
ax.set_xticks(ticks=[time_index.get_loc(t) for t in monthly_ticks])
ax.set_xticklabels(monthly_ticks_str, rotation=45, ha='right')

# Set the y-ticks on the right side
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax.set_yticks(ticks=np.arange(len(ordered_row_names)), labels=ordered_row_names, fontsize=8, rotation=0)
ax.set_yticklabels(ordered_row_names, fontsize=font_size, rotation=0)

# Set labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Paddock / Crop Type')
ax.set_title('Fractional Cover')
ax.grid(False)

# Save the plot to results
filename = outdir + stub + f"_pt-{'fractional'}.png"
plt.savefig(filename, bbox_inches='tight')
print(filename)
# -

# ## ANOVA

# +
# ANOVA and box plot of fractional cover
# I'm using the df and the rgb_image_ordered as these were the final outputs to go into the heatmap above

# Setup a dictionary to store the values for each category
fractional_score = dict()
for type in df['type'].unique():
    fractional_score[type] = []
    
# Sum the bare ground values for each paddock
for rgb_time_series, type in zip(rgb_image_ordered, df['type']):
    bare = rgb_time_series[:,0]
    bare_score = bare.sum()
    fractional_score[type].append(bare_score)

fractional_score.keys()
# -

# Comparing named and cropping paddocks to unnamed paddocks
F_statistic, p_value = stats.f_oneway(fractional_score['named'] + fractional_score['cropping'], fractional_score['unnamed'])
print("Comparing named and unnamed paddocks: P-value =", p_value)

# Comparing cropping paddocks to pasture paddocks
F_statistic, p_value = stats.f_oneway(fractional_score['cropping'], fractional_score['named'])
print("Comparing cropping and pasture paddocks: P-value =", p_value)

# Comparing forests and trees to paddocks
F_statistic, p_value = stats.f_oneway(fractional_score['forest'] + fractional_score['tree_row'], 
                                      fractional_score['named'] + fractional_score['cropping'] + fractional_score['unnamed'])
print("Comparing paddocks and tree-rows/forests: P-value =", p_value)

# Comparing the means
means = {type:np.mean(scores) for type, scores in fractional_score.items()}
print("Mean score: ", means)

# ## Box Plots

# +
# Comparing everything
fig, ax = plt.subplots(figsize=(8, 6))

# Create the plot with different colors for each group
boxplot = ax.boxplot(x=[scores for type, scores in fractional_score.items()],
                     labels=fractional_score.keys(),
                     patch_artist=True,
                     medianprops={'color': 'black'}
                    ) 
plt.title("Total bare ground in all categories")
plt.show()
# -

named = fractional_score['named'] + fractional_score['cropping']
unnamed = fractional_score['unnamed']

# +
# Comparing unnamed vs named
fig, ax = plt.subplots(figsize=(8, 6))

named = fractional_score['named'] + fractional_score['cropping']
unnamed = fractional_score['unnamed']

# Create the plot with different colors for each group
boxplot = ax.boxplot(x=[named, unnamed],
                     labels=["named", "unnamed"],
                     patch_artist=True,
                     medianprops={'color': 'black'}
                    ) 
plt.title("Total bare ground")
plt.show()

# +
# Comparing cropping vs pasture
fig, ax = plt.subplots(figsize=(8, 6))

# Create the plot with different colors for each group
boxplot = ax.boxplot(x=[fractional_score['cropping'], fractional_score['named']],
                     labels=["cropping", "pasture"],
                     patch_artist=True,
                     medianprops={'color': 'black'}
                    ) 
plt.title("Total bare ground")
plt.show()

# +
# Comparing paddocks vs tree rows/forest blocks
fig, ax = plt.subplots(figsize=(8, 6))

trees = fractional_score['forest'] + fractional_score['tree_row']
paddocks = fractional_score['named'] + fractional_score['cropping'] + fractional_score['unnamed']

# Create the plot with different colors for each group
boxplot = ax.boxplot(x=[trees, paddocks],
                     labels=["trees", "paddocks"],
                     patch_artist=True,
                     medianprops={'color': 'black'}
                    ) 
plt.title("Total bare ground")
plt.show()
