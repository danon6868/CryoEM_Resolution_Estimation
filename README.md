# Deep learning approach for resolution estimation of cryoelectron microscopy data

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

To train the model on experimental data, 180 protein structures with different resolution and molecular weight were selected. This structures were obtained using cryoelectron microscopy. Next, for each electron density map (simulated or experimental), a local resolution map was obtained using the Resmap program with default parameters. After that, the simulated and experimental data were independently divided into three parts: training, validation and test in the ratio 0.7:0.15:0.15.
Further data processing was as follows: all electron density maps and their corresponding local resolution maps were divided into small fragments of 16x16x16 voxels (it was a hyperparameter and was tuned using train/test split). This was done, on the one hand, so that the model had the opportunity to look at some area around each element of the volume to assess the resolution in it, but on the other hand, so that these fragments were small enough. The values ​​of the electron density maps were clipped to the range from 0 to 1 using min-max normalization.

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

+ Run model training and/or inference on **Google Colab** via [link](https://colab.research.google.com/github/danon6868/CryoEM_Resolution_Estimation/blob/main/3D_UNet_resolution_estimation.ipynb)
+ Run model training and/or inference locally. Here you can also run scripts for data preparation (but addional tools required)

### Launch notebook in Google Colab

Follow [this](https://colab.research.google.com/github/danon6868/CryoEM_Resolution_Estimation/blob/main/3D_UNet_resolution_estimation.ipynb) link.


### Local running

You need `git` to be installed. Open terminal (`Crtl+Alt+t`) and run following commands:

```bash
git clone https://github.com/danon6868/CryoEM_Resolution_Estimation.git
cd CryoEM_Resolution_Estimation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then you should install `PyTorch`. It is more difficult than installing other libraries, but [this](https://varhowto.com/install-pytorch-ubuntu-20-04/) post will help to understand even a beginner. If you have `GPU` on your computer you can install `PyTorch` with `CUDA` support else you can install `PyTorch` with `CPU` support only.

After all required libraries installation you can use all supported functions. If you don't want to create your own dataset or retrain model, you can go to the point `5`. **To run all scripts you should be in** `CryoEM_Resolution_Estimation` **directory**.

#### **1. Generate electron density maps based on **.pdb** files**

If you want to generate electron density maps from raw **.pdb** files, run following commands in terminal (you should be in `CryoEM_Resolution_Estimation` directory):

```bash
mkdir data
cd data
mkdir pdb_files
mkdir mrc_for_pdb_files
```

Then put chosen **.pdb** files in `pdb_files` directory. `mrc_for_pdb_files` will contain generated **.mrc** files.
You need `EMAN2` to be installed. Consult [this](https://blake.bcm.edu/emanwiki/EMAN2/Install) page to know more about `EMAN2` installation.
Then open terminal and run `generate_maps.py` script:

```bash
python data_preparation/generate_maps.py
```

When scripts finishes, the generated electron density maps will be in `data/mrc_for_pdb_files` directory.

#### **2. Run Resmap to create targets for 3D-UNet**

Local resolution maps are targets for 3D-Unet. If you want create your own dataset from electron density **.mrc** files, you should install `Resmap` locally (consult [this](http://resmap.sourceforge.net/) link). Then run `run_resmap.py` script:

```bash
python data_preparation/run_resmap.py
```

Local resolution maps will be in `data/targets` directory.

#### **3. Data processing**

To create dataset for model training and validation you need 2 directories:

+ `data/mrc_for_pdb_files` &mdash; with electron density maps
+ `data/targets` &mdash; with local resolution maps

Then run `process_data.py`:

```bash
python data_preparation/process_data.py
```

#### **4. Train model**

To train model on custom dataset:

```
usage: train_model.py [-h] [--train_data TRAIN_DATA] [--valid_data VALID_DATA] [--n_epoches N_EPOCHES] [-v {0,1,2}] [--out_weights_dir OUT_WEIGHTS_DIR]
                      [--out_weights_filename OUT_WEIGHTS_FILENAME]

optional arguments:
  -h, --help            show this help message and exit
  --train_data TRAIN_DATA
                        Path to file with train samples.
  --valid_data VALID_DATA
                        Path to file with validation samples
  --n_epoches N_EPOCHES
                        The number of epoches to train the model
  -v {0,1,2}, --verbose {0,1,2}
                        The higher the value, the more information about training will be displayed to the user
  --out_weights_dir OUT_WEIGHTS_DIR
                        The directory where model weights will be saved
  --out_weights_filename OUT_WEIGHTS_FILENAME
                        The filename with trained model weights
```

**Example:**

In this trial 3D-Unet will be trained during 30 epochs using hyperparameters from `resolution_estimation_with_dl/model_training/train_config.py` on example train data from `resolution_estimation_with_dl/example_data/example_train_data.hdf5`. For simplicity, validation will pass on the same data.

```bash
python -m resolution_estimation_with_dl.model_training.train_model --train_data resolution_estimation_with_dl/example_data/example_train_data.hdf5 --valid_data resolution_estimation_with_dl/example_data/example_train_data.hdf5 --n_epochs 30
```

Script will create `resolution_estimation_with_dl/model_weights` directory with `unet_3d_trained_weights.pth` model weights inside.

#### **5. Run model**

To estimate local resolution map for one **.mrc** file:

```
usage: run_model.py [-h] [--models_path MODELS_PATH] [--model_name {unet_3d_trained_dropout.pth,unet_3d_trained_batchnorm.pth}] [--electron_density_map ELECTRON_DENSITY_MAP]
                    [--output_file_name OUTPUT_FILE_NAME] [--device {cpu,cuda:0}]

optional arguments:
  -h, --help            show this help message and exit
  --models_path MODELS_PATH
                        Path to directory with saved models
  --model_name {unet_3d_trained_dropout.pth,unet_3d_trained_batchnorm.pth}
                        Model name to load. `unet_3d_trained_dropout.pth` and `unet_3d_trained_batchnorm.pth` are available and should be in `models_path` directory or they will be downloaded (it
                        is about 400 Mb). If you want to use your own model, you should put its weights into `models_path` directory yourself. Experiments shown that model with dropout was better
                        on our experimental data
  --electron_density_map ELECTRON_DENSITY_MAP
                        Electron density map for which you want to estimate local resolution map. By default it will be example map
  --output_file_name OUTPUT_FILE_NAME
                        File name for local resolution map. By default it will be a `electron_density_map` name with `resulotion` suffix
  --device {cpu,cuda:0}
                        Which device to use for model inference. If nothing was given, device from `train_config.py` will be used
```

**Example:**

In this example we will use 3D-UNet to estimate local resolution map for [this](https://www.ebi.ac.uk/emdb/EMD-13939) electron density map. To run model pretrained weights should be in `resolution_estimation_with_dl/model_weights` directory. Script `run_model.py` will download chosen weights if ones are missing in this directory. Following command will estimate local resolution map for EMD-13393 using `CPU` as device:

```bash
python -m resolution_estimation_with_dl.resolution_estimation.run_model --electron_density_map resolution_estimation_with_dl/example_data/13939_map.mrc --model_name unet_3d_trained_dropout.pth --device cpu
```

Padded electron density map (for visualization) and estimated local resolution map will be in `results` directory. Then you can you use any tool for **.mrc** files visualization, e.g. `Chimera`. On a figure below you can see EMD-13939 coloured by local resolution value:

![img](/images/model_example_13939.png)

## Contacts

If you have any questions, please contact @Danil_litvinov &mdash; Telegram or danon6868@gmail.com &mdash; Email.
