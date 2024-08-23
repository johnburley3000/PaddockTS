# +
# Idea of this notebook is assign productivity and shelter scores and plot them against each other

# +
# Standard Libraries
import pickle

# Dependencies
import xarray as xr

# Local imports
os.chdir(os.path.join(os.path.expanduser('~'), "Projects/PaddockTS"))

# -





filename = "/g/data/xe2/John/Data/PadSeg/MILGA_ds2.pkl"

with open(filename, 'rb') as file:
    ds = pickle.load(file)

ds


