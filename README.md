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

The image below shows a comparison of the work of the trained model and similar tools. It can be seen that the neural network produces predictions that generally agree with the results of existing analogues, and often the results of the work are intermediate between the results of Resmap and Monores, although training took place only on data obtained using Resmap. 

![img](/images/compare_with_other_tools.png)

## Data description

To train the model on experimental data, 180 protein stuctures with different resolution and molecular weight were selected. This structures were obtained using cryoelectron microscopy. Next, for each electron density map (simulated or experimental), a local resolution map was obtained using the Resmap program (Kucukelbir, Sigworth, Tagare, 2014) with default parameters. After that, the simulated and experimental data were independently divided into three parts: training, validation and test in the ratio 0.7:0.15:0.15.
Further data processing was as follows: all electron density maps and their corresponding local resolution maps were divided into small fragments of 16x16x16 voxels (it was a hyperparameter and was tuned using train/test split). This was done, on the one hand, so that the model had the opportunity to look at some area of each element of the volume to assess the resolution in it, but on the other hand, so that these fragments were small enough. The values ​​of the electron density maps were converted to the range from 0 to 1 using min-max normalization.

In `resolution_estimation_with_dl/example_data` directory you can find several files, which will help you to get acquainted with this work:

+ `3l08.pdb` &mdash; [protein structure](https://www.rcsb.org/structure/3L08) for electron density map simulation
+ `13939_map.mrc` &mdash; [electron density map](https://www.ebi.ac.uk/emdb/EMD-13939) for running Resmap and obtaining local resolution map
+ `example_train_data.hdf5` &mdash; small dataset which can be used to run model training

## Workflow overview

All code is stored in `resolution_estimation_with_dl` package.

### Data preparation

Code for data processing is stored in `resolution_estimation_with_dl/data_preparation`:

+ `generate_maps.py` &mdash; this code was used to generate electron density maps from raw **.pdb** files using `pdb2mrc` function from **EMAN2** package
+ `run_resmap.py` &mdash; this code was used to estimate local resolution map for all **.mrc** files
+ `data_prep_utils.py` &mdash; functions for processing pairs: electron density and local resolution maps
+ `process_data.py` &mdash; script for data processing

### Model training

Code for model training is stored in `resolution_estimation_with_dl/model_training`:

+ `training_config.py` &mdash; this file contains different hyperparameters for model training, e.g. *learning rate*
+ `model.py` &mdash; class of 3D-UNet model
+ `training_utils.py` &mdash; classes and functions for model training
+ `train_model.py` &mdash; script that runs model training

### Resolution estimation

Code for resolution estimation for a given electron density map is stored in `resolution_estimation_with_dl/resolution_estimation`:

+ `utils.py` &mdash; functions for model inference (local resolution estimation)
+ `run_model.py` &mdash; script that runs model inference

## Usage

All code was run on **python 3.9.7** on **Ubuntu 22.04** and **Ubuntu 20.04**. Also functionality was tested on **Google Colab**. Correct work on other vesrions is not guaranteed.

There are several options for working:

+ Run model inference on **Google Colab**


![img](/images/model_example_13939.png)





