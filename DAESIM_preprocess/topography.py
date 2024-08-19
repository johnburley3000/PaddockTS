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
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)

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
    branches = grid.extract_river_network(fdir, acc > np.max(acc)/(num_catchments*10), dirmap=dirmap)

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
        catch = grid.catchment(x=x, y=y, fdir=fdir, dirmap=dirmap, 
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
    plt.title('Flow Accumulation', size=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()


show_acc(acc)

# +
fig = plt.figure(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(fdir, extent=grid.extent, cmap='viridis', zorder=2)
boundaries = ([0] + sorted(list(dirmap)))
plt.colorbar(boundaries= boundaries,
             values=sorted(dirmap))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flow direction grid', size=14)
plt.grid(zorder=-1)
plt.tight_layout()
# -

# %%time
num_catchments = 10
gullies, full_branches = catchment_gullies(grid, fdir, acc, num_catchments)
ridges = catchment_ridges(grid, fdir, acc, full_branches)


# +
def show_contour_overlays(dem, overlays, colours, title=""):
    """Plot an image of the contours and the overlay layer"""
    plt.imshow(dem, alpha=0)
    plt.contour(dem, levels=10, alpha=0.5)
    for overlay, colour in zip(overlays, colours):
        y, x = np.where(overlay)
        plt.scatter(x, y, marker='.', s=0.5, c=colour)
    plt.title(title)
    plt.show()

show_contour_overlays(dem, [gullies, ridges], ["blue", "red"], "Ridges and Gullies")
