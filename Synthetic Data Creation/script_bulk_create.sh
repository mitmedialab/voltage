#!/bin/bash
ctr=0
MAX_NEURONS=26
TO=100
for filename in ../base_data/data/*.tif; do
    ctr=$(( $ctr + 1 ))
    echo "Creating dataset $ctr"
    timeout $TO python create_synthetic_data_ver10.py $ctr $MAX_NEURONS $filename
    if [ $? -ne 0 ]; then
        echo "***************** ERRORRRRRRR *******************"
    fi
done