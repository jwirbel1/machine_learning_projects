#!/bin/bash
PATH1 = /dmc/ml_storage/machine_learning/Final_ML_CBS/data/train
PATH2 = /dmc/ml_storage/machine_learning/Final_ML_CBS/data/train/0
for file in $(cat image_ids.csv); do 
    # copy the file to the new directory
    cp $PATH1/$file $PATH2
done