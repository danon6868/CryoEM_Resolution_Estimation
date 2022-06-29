# Deep learning approach for resolution estimation of cryoelectron microscopy data

This repository will contain source code for cryoEM maps resolution estimation using neural networks.

## Purpose and objectives

### Purpose

+ Develop an algorithm, which can take an electron density map (from cryoEM) and estimate local resolution map.

### Objectives

+ Collect and prepare training data
+ Develop and train neural network model that can estimate local resolution map based on electron density map
+ Select optimal model architecture, training hyperparameters and data processing methods
+ Validate obtained model on my own experimental data
+ Based on the trained model, create tool for estimation local resolution maps

## Results

During this work several models for local resolution map estimation were designed. The best one is 3D-UNet model with Dropout.

![img](/images/model_architecture.png)

On the figure above there is architecture of 3D-UNet model with Dropout. Classic UNet commonly used for image segmentation task and it is essentially a classification (e. g. object vs background). But in this case we used this model for regression, because we want to predict real numbers.

In the course of this work, some models were trained on simulated data, but these models showed poor results on experimental data and it was decided to abandon this approach. Despite the fact that the model trained more slowly on the experimental data and at the end the MAE reached about 0.5-0.6 angstroms (0.3 on the simulated data), this model showed more adequate results on test data. On the figure below you can see learning curves of model that was trained on experimental data.

![img](/images/experimental_data_metric.png)


## Data description

## Data collection and processing

### Simulated data

### Experimental data

## Model building and training

## Test model





