import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
import pickle
import xarray as xr
import rioxarray  # activate the rio accessor

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""
Obtain paddock-level time series reflectance for pre-segmented paddocks

Example usage:
python3 Code/03_paddock-ts.py --stub test1 --outdir /g/data/xe2/John/Data/PadSeg/

Assumes that paddocks shapefile (*filt.gpkg) and sentinel xarray dataset *ds2.pkl exist in the outdir
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--stub", type=str, required=True, help="Stub name for file naming")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for saved files")
    return parser.parse_args()

def drop_oa(ds):
    """Subset the list of variables in an xarray object to then filter it"""
    l = list(ds.keys())
    l = [item for item in l if 'oa' not in item]
    print('Keeping vars:', l)
    print('Number of variables:', len(l))
    return l

def main():
    args = parse_arguments()
    stub = args.stub
    outdir = args.outdir

    # read in the polygons and add the paddock column:
    pol = gpd.read_file(outdir + stub + '_filt.gpkg')
    pol['paddock'] = range(1, len(pol) + 1)
    pol['paddock'] = pol.paddock.astype('category')

    ## Open the satellite data stack
    with open(outdir + stub + '_ds2.pkl', 'rb') as handle:
        ds = pickle.load(handle)

    keep_vars = drop_oa(ds)

    # Make paddock-variable-time (pvt) array:
    ts = []
    for datarow in pol.itertuples(index=True):
        ds_ = ds[keep_vars]
        ds_clipped = ds_.rio.clip([datarow.geometry])
        pol_ts = ds_clipped.where(ds_clipped > 0).median(dim=['x', 'y'])
        array = pol_ts.to_array().transpose('variable', 'time').values.astype(np.float32)
        ts.append(array[None, :])
    pvt = np.vstack(ts)

    # save the pvt
    np.save(outdir + stub + '_pvt', pvt, allow_pickle=True, fix_imports=True)

    # save the list of variable names:
    with open(outdir + stub + '_pvt_vars.pkl', 'wb') as f:
        pickle.dump(keep_vars, f)

    print('Created the file: ', outdir + stub + '_pvt_vars.pkl')
    print('Finished!')

if __name__ == "__main__":
    main()

