# machine_learning_projects
This repository contains code for the machine leanring projects I undertook during my master studies at CBS and ITU.

## [KAN-CDSCO1004U](https://github.com/jwirbel1/machine_learning_projects/tree/main/KAN-CDSCO1004U)
This course was part of my master studies and the code contained is for the final group handin, in which we scored a 12 out of 12. The code in this repository contains two CNN-Models based on the VGG-16 archietecture with the goal of predicting the age cohort and gender of a patient from a chest X-Ray. The dataset is the SPR X-Ray Age and Gender Dataset (10.34740/kaggle/dsv/5142976), which was preprocessed by us to get the best image classification results.
### Models
1) single-model: this model based on the VGG-16 architecture predicts the age cohort and gender of an given image with one single classifier
2) dual-model: also based on the VGG-16 architecture, this model uses two separate classifiers to predict the age cohort and gender.
3) feature-extractor: this feature extractor was used in combination with an SVM to create our best predictions for the dataset.

