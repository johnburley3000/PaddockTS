# +
# Pysheds documentation is here: https://mattbartos.com/pysheds/

# +
import os

# Dependencies
import numpy as np
from scipy import ndimage
from pysheds.grid import Grid
import matplotlib.pyplot as plt
from matplotlib import colors

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))
from DAESIM_preprocess.util import gdata_dir, scratch_dir

# +
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

def pysheds_accumulation(tiff_file):
    """Read in the grid and dem and calculate the water flow direction and accumulation"""
    
    # Load both the dem (basically a numpy array), and the grid (all the metadata like the extent)
    grid = Grid.from_raster(tiff_file)
    dem = grid.read_raster(tiff_file)

    # Hydrologically enforce the DEM so water can flow downhill to the edge and not get stuck
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Calculate the direction and accumulation of water
    fdir = grid.flowdir(inflated_dem)
    acc = grid.accumulation(fdir)

    return grid, dem, fdir, acc

def find_segment_above(acc, coord, branches_np):
    """Look for a segment upstream with the highest accumulation"""
    segment_above = None
    acc_above = -1
    for i, branch in enumerate(branches_np):
        if branch[-1] == coord:
            branch_acc = acc[branch[-2][0], branch[-2][1]] 
            if branch_acc > acc_above:
                segment_above = i
                acc_above = branch_acc
    return segment_above

def catchment_gullies(grid, fdir, acc, num_catchments=10):
    """Find the largest gullies"""

    # Extract the branches
    branches = grid.extract_river_network(fdir, acc > np.max(acc)/(num_catchments*10))

    # Convert the branches to numpy coordinates 
    branches_np = []
    for i, feature in enumerate(branches["features"]):
        line_coords = feature['geometry']['coordinates']
        branch_np = []
        for coord in line_coords:
            col, row = ~grid.affine * (coord[0], coord[1])
            row, col = int(round(row)), int(round(col))
            branch_np.append([row,col])
        branches_np.append(branch_np)

    # Repeatedly find the main segments to the branch with the highest accumulation. 
    full_branches = []
    for i in range(num_catchments):
        
        # Using the second last pixel before it's merged with another branch.
        branch_accs = [acc[branch[-2][0], branch[-2][1]] for branch in branches_np]
        largest_branch = np.argmax(branch_accs)

        # Follow the stream all the way up this branch
        branch_segment_ids = []
        while largest_branch != None:
            upper_coord = branches_np[largest_branch][0]
            branch_segment_ids.append(largest_branch)
            largest_branch = find_segment_above(acc, upper_coord, branches_np)

        # Combine the segments in this branch
        branch_segments = [branches_np[i] for i in sorted(branch_segment_ids)]
        branch_combined = [item for sublist in branch_segments for item in sublist]
        full_branches.append(branch_combined)

        # Remove all the segments from that branch and start again
        branch_segments_sorted = sorted(branch_segment_ids, reverse=True)
        for i in branch_segments_sorted:
            del branches_np[i]

    # Extract the gullies
    gullies = np.zeros(acc.shape, dtype=bool)
    for branch in full_branches:
        for x, y in branch:
            gullies[x, y] = True

    return gullies, full_branches


def catchment_ridges(grid, fdir, acc, full_branches):
    """Finds the ridges/catchment boundaries corresponding to those gullies"""

    # Progressively delineate each catchment
    catchment_id = 1
    all_catchments = np.zeros(acc.shape, dtype=int)
    for branch in full_branches:
        
        # Find the coordinate with second highest accumulation
        coords = branch[-2]

        # Convert from numpy coordinate to geographic coordinate
        x, y = grid.affine * (coords[1], coords[0])

        # Generate the catchment above that pixel
        catch = grid.catchment(x=x, y=y, fdir=fdir, 
                            xytype='coordinate')

        # Override relevant pixels in all_catchments with this new catchment_id
        all_catchments[catch] = catchment_id
        catchment_id += 1

    # Find the edges of the catchments
    sobel_x = ndimage.sobel(all_catchments, axis=0)
    sobel_y = ndimage.sobel(all_catchments, axis=1)  
    edges = np.hypot(sobel_x, sobel_y) 
    ridges = edges > 0

    return ridges



# -

# !ls /g/data/xe2/cb8590/Data/PadSeg/*_terrain_cleaned.tif

filepath = "/g/data/xe2/cb8590/Data/PadSeg/MILG_6km_terrain_cleaned.tif"

# %%time
grid, dem, fdir, acc = pysheds_accumulation(filepath)


# +
def show_acc(acc):
    """Very pretty visualisation of water accumulation"""
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)
    plt.grid('on', zorder=0)
    im = ax.imshow(acc, zorder=2,
                   cmap='cubehelix',
                   norm=colors.LogNorm(1, acc.max()),
                   interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Upstream Cells')
    plt.title('Topographic Index', size=14)
    plt.tight_layout()
    plt.show()
    
show_acc(acc)
# -

# %%time
num_catchments = 10
gullies, full_branches = catchment_gullies(grid, fdir, acc, num_catchments)
ridges = catchment_ridges(grid, fdir, acc, full_branches)


# +
def show_ridge_gullies(dem, ridges, gullies):
    """Very pretty visualisation of ridges and gullies"""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0)
    
    # Plot the DEM
    im = ax.imshow(dem, cmap='terrain', zorder=1, interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Elevation (m)')
    
    # Overlay ridges and gullies
    ax.contour(ridges, levels=[0.5], colors='red', linewidths=1.5, zorder=2)
    ax.contour(gullies, levels=[0.5], colors='blue', linewidths=1.5, zorder=3)
    ax.contour(dem, colors='black', linewidths=0.5, zorder=4, alpha=0.5)

    plt.title('Ridges and Gullies', size=14)
    plt.tight_layout()
    plt.show()

show_ridge_gullies(dem, ridges, gullies)


# +
def show_aspect(fdir):
    """Somewhat pretty visualisation of the aspect"""
    
    # Apparently these are the default ESRI directions
    directions = {
        64: "North",
        128: "Northeast",
        1: "East",
        2: "Southeast",
        4: "South",
        8: "Southwest",
        16: "West",
        32: "Northwest",
        -1: "Flat"
    }
    
    # I arrived at these colours through trial and error
    colours = ['#808080',  # Grey
               '#EE82EE',  # Violet
               '#00008B',  # Dark Blue
               '#ADD8E6',  # Light Blue
               '#006400',  # Dark Green
               '#90EE90',  # Light Green
               '#FFFF00',  # Yellow
               '#FFA500',  # Orange
              ]
    cmap = mcolors.ListedColormap(colours)
    bounds = sorted(list(directions.keys()))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Plotting the aspect 
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(fdir, cmap=cmap, norm=norm, zorder=2)
    cbar = plt.colorbar(im, ticks=sorted(directions.keys()))
    cbar.ax.set_yticklabels([directions[key] for key in sorted(directions.keys())])
    plt.title('Aspect', size=14)
    plt.tight_layout()

show_aspect(fdir)
