#!/bin/bash

# iterate over the folder /dmc/ml_storage/machine_learning/Final_ML_CBS/data/age_csvs and read the file contents
for file in /dmc/ml_storage/machine_learning/Final_ML_CBS/data/age_csvs/*.csv
do
        # get the filename without the extension
        filename=$(basename "$file" .csv)

        mkdir /dmc/ml_storage/machine_learning/Final_ML_CBS/data/age/$filename
        cat $file | while read line
        do
        # copy the image with the filename to the folder with the filename
                echo $line
                cp /dmc/ml_storage/machine_learning/Final_ML_CBS/data/train/$line /dmc/ml_storage/machine_learning/Final_ML_CBS/data/age/$filename
    done
done