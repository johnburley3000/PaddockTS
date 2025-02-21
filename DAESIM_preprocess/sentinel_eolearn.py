# !pip install eo-learn

# +
# Useful links:
# Example notebook: https://github.com/sentinel-hub/eo-learn/blob/master/examples/io/SentinelHubIO.ipynb

# +
# %matplotlib inline
import datetime
import json

import matplotlib.pyplot as plt
from aenum import MultiValueEnum
from matplotlib.colors import BoundaryNorm, ListedColormap

from sentinelhub import CRS, BBox, DataCollection, SHConfig

from eolearn.core import EOWorkflow, FeatureType, LoadTask, OutputTask, SaveTask, linearly_connect_tasks
from eolearn.io import SentinelHubDemTask, SentinelHubEvalscriptTask, SentinelHubInputTask

# +
# Load your Copernicus Client ID and Secret from a json file that doesn't get committed to the repository
with open("credentials.json", "r") as file:
    credentials = json.load(file)
    
CLIENT_ID = credentials["CLIENT_ID"]
CLIENT_SECRET = credentials["CLIENT_SECRET"]

print("Client ID:", CLIENT_ID)
print("Client Secret:", CLIENT_SECRET) 

# +
config = SHConfig()
config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET
config.instance_id = 'chris_test'

# config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
# config.sh_base_url = "https://sh.dataspace.copernicus.eu"
# -

if config.sh_client_id == "" or config.sh_client_secret == "" or config.instance_id == "":
    print("Warning! To use Sentinel Hub services, please provide the credentials (client ID and client secret).")

# +
# region of interest
roi_bbox = BBox(bbox=[5.60, 52.68, 5.75, 52.63], crs=CRS.WGS84)

# time interval of downloaded data
time_interval = ("2018-04-01", "2018-05-01")

# maximal cloud coverage (based on Sentinel-2 provided tile metadata)
maxcc = 0.8

# resolution of the request (in metres)
resolution = 20

# time difference parameter (minimum allowed time difference; if two observations are closer than this,
# they will be mosaicked into one observation)
time_difference = datetime.timedelta(hours=2)
# -

input_task = SentinelHubInputTask(
    data_collection=DataCollection.SENTINEL2_L1C,
    bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
    bands_feature=(FeatureType.DATA, "L1C_data"),
    additional_data=[(FeatureType.MASK, "dataMask")],
    resolution=resolution,
    maxcc=maxcc,
    time_difference=time_difference,
    config=config,
    max_threads=3,
)

indices_evalscript = """
    //VERSION=3

    function setup() {
        return {
            input: ["B03","B04","B08","dataMask"],
            output:[{
                id: "indices",
                bands: 2,
                sampleType: SampleType.FLOAT32
            }]
        }
    }

    function evaluatePixel(sample) {
        let ndvi = index(sample.B08, sample.B04);
        let ndwi = index(sample.B03, sample.B08);
        return {
           indices: [ndvi, ndwi]
        };
    }
"""

# this will add two indices: ndvi and ndwi
add_indices = SentinelHubEvalscriptTask(
    features=[(FeatureType.DATA, "indices")],
    evalscript=indices_evalscript,
    data_collection=DataCollection.SENTINEL2_L1C,
    resolution=resolution,
    maxcc=maxcc,
    time_difference=time_difference,
    config=config,
    max_threads=3,
)

add_dem = SentinelHubDemTask(
    feature="dem", data_collection=DataCollection.DEM_COPERNICUS_30, resolution=resolution, config=config
)

add_l2a_and_scl = SentinelHubInputTask(
    data_collection=DataCollection.SENTINEL2_L2A,
    bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
    bands_feature=(FeatureType.DATA, "L2A_data"),
    additional_data=[(FeatureType.MASK, "SCL")],
    resolution=resolution,
    maxcc=maxcc,
    time_difference=time_difference,
    config=config,
    max_threads=3,
)

save = SaveTask("io_example")


output_task = OutputTask("eopatch")


# +
workflow_nodes = linearly_connect_tasks(input_task, add_indices, add_l2a_and_scl, add_dem, save, output_task)
workflow = EOWorkflow(workflow_nodes)

result = workflow.execute(
    {
        workflow_nodes[0]: {"bbox": roi_bbox, "time_interval": time_interval},
        workflow_nodes[-2]: {"eopatch_folder": "eopatch"},
    }
)
# -


