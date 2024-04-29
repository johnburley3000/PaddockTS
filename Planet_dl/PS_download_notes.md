#Coding notes to "semi"-automate Planet downloads. Convert to sbatch later

###Goal:
For a given ROI, download all of the Planet data from PSB.SD (and other sensors?)
Get all available bands for each sensor. 
Clip the downloaded data to the ROI

###Challenge:
Because there's a limit of 500 items per order. Need to split a list over several orders. 

Need to have planet CLI installed and activated with user Key

### Variables:
ID=new1 # name of the ROI. At this stage, assumes that a .geojson and _clip.json exist. Ideally, generate these from coords and buffer distance 

### STEP 1: 
Generate a list of all product IDs, then split it by 500 

planet data filter \
--permission \
--std-quality \
--date-range acquired gte 2017-10-01 \
--date-range acquired lt 2024-03-01 \
--range clear_percent gt 90 \
--string-in instrument PSB.SD \
--geom ${ID}.geojson \
| planet data search \
--limit 0 \
PSScene \
--filter - \
| jq -r .id > item_list.txt

rm x*
cat item_list.txt | split --lines=500 -d
# this makes files called x00, x01, x02, ... (remove old ones first)

## STEP 2 Create the orders. 

for i in x*
do

unique name
order_name=${ID}_$i

convert product ids to comma separated
ids=$(cat $i | tr "\n" "," | rev | cut -c2- | rev) 

create order for visual data
planet orders request \
--item-type PSScene \
--bundle visual \
--name ${order_name}_Vis \
$ids \
--clip ${ID}_clip.json \
--pretty \
--email \
| planet orders create -

create order for 8band data
planet orders request \
--item-type PSScene \
--bundle analytic_8b_sr_udm2 \
--name ${order_name}_8b \
$ids \
--clip ${ID}_clip.json \
--pretty \
--email \
| planet orders create -
# one option here is to combine the json files for each order.. but this might not be worth the effort

done


## STEP 3 download the data
For each list wait for order to be ready, then download

set the directory to dump into, based on the data:
DIR=/g/data/xe2/datasets/Planet/Trees/$ID
mkdir $DIR

get the unique order ids
orders=$(planet orders list | jq -rs '.[] | "\(.id) \(.name)"' | grep "$ID" | awk '{print $1}')
orders=$(planet orders list | jq -rs '.[] | "\(.id) \(.name)"' | grep "$ID" | grep "Vis" | awk '{print $1}')

for each order id, wait, then download:
### MIGHT NEED TO BE RUN REMOTELY IN CASE IT TIMES OUT
for order_id in $orders
do

planet orders wait \
--max-attempts 0 \
$order_id \
&& planet orders download \
--checksum MD5 \
--directory $DIR \
$order_id

echo 'FINISHED DOWNLOAD FOR: '$order_id

done

# Should be able to remove the --overwrite flag if some data already downloaded to save the quota... 

## FINISHED!

# This should drop all the data into a directory for the ROI (i.e. ID), with one sub-directory per order_id.

# Quesions:
-> optimal zipping?
-> obtaining multiple bundles?
-> obtaining multiple item types (i.e. Landsat) given that the same flags aren't recognized.
-> combining order numbers?
-> how to automatically generate jsons for search and clip using coordinates and buffer?
-> is there a way to mosaic imagery for each day so that I avoid doubleups?


### STEP 3 workaround for "wait" not working (don't use this...)

for order_id in $orders
do

planet orders download \
--checksum MD5 \
--directory $DIR \
--overwrite \
$order_id

done
	
	
