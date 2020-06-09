# GRU based network to predict temporal distribution of streamflow downstream of the basin

Accompanying code for our "Streamflow Forecasting with Deep Learning Methods" project for CS230 Spring 2020 class. This code is built upon the code repository [2]. Contributors: Rahul Dhakal, Josue Fonseca, Rohan Pasalkar

## Content of the repository

- `main.py` Main python file used for training and evaluating of our models, as well as to perform the robustness analysis
- `data/` contains the list of basins (USGS gauge ids) considered in our study
- `papercode/` contains the entire code (beside the in the root directory `main.py` file)
- `papercode/gru.py` contains the code for A-GRU deep neural network.

## Setup to run the code locally

Download this repository either as zip-file or clone it to your local file system by running

```
git clone https://github.com/rrdhakal/Streamflow-Forecasting--Deep-Learning-Method-Stanford-CS-230-.git
```

### Setup Python environment
Within this repository we provide two environment files (`environment_cpu.yml` and `environment_gpu.yml`) that can be used with Anaconda or Miniconda to create an environment with all packages needed.

Simply run

```
conda env create -f environment_cpu.yml
```
for the cpu-only version. Or run

```
conda env create -f environment_gpu.yml
```
if you have a CUDA capable NVIDIA GPU. This is recommended if you want to train/evaluate the models on you machine but not strictly necessary. 

However, it is not strictly needed to re-train or re-evaluate any of the models to run the notebooks. Just make sure to download our pre-trained models and the pre-calculated model evaluations.

## Data needed

### Required Downloads

First of all you need the CAMELS data set, to run any of your code. This data set can be downloaded for free here:

- [CAMELS: Catchment Attributes and Meteorology for Large-sample Studies - Dataset Downloads](https://ral.ucar.edu/solutions/products/camels) Make sure to download the `CAMELS time series meteorology, observed flow, meta data (.zip)` file, as well as the `CAMELS Attributes (.zip)`. Extract the data set on your file system and make sure to put the attribute folder (`camels_attributes_v2.0`) inside the CAMELS main directory.

However, we trained our models with an updated version of the Maurer forcing data, that is still not published officially (CAMELS data set will be updated soon). The updated Maurer forcing contain daily minimum and maximum temperature. The original Maurer data included in the CAMELS data set only includes daily mean temperature. You can find the updated forcings temporarily here:

- [Updated Maurer forcing with daily minimum and maximum temperature](https://www.hydroshare.org/resource/17c896843cf940339c3c3496d0c1c077/)

Download and extract the updated forcing into the `basin_mean_forcing` folder of the CAMELS data set and do not rename it (name should be `maurer_extended`).

Next you need the simulations of all benchmark models. These can be downloaded from HydroShare under the following link:

- [CAMELS benchmark models](http://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1)

## Run locally

For training or evaluating any of the models a CUDA capable NVIDIA GPU is recommended but not strictly necessary. Since we only train/use LSTM-based models a strong, multi-core CPU will work as well.

Before starting to do anything, make sure you have activated the conda environment.

```
conda activate ealstm
```

### Train model
To train a model, run the following line of code from the terminal

```
python main.py train --camels_root /path/to/CAMELS
```
This would train a single A-GRU model with a randomly generated seed using the basin average NSE as loss function and store the results under `runs/`. Additionally the following options can be passed:

- `--seed NUMBER` Train a model using a fixed random seed
- `--cache_data True` Load the entire training data into memory. This will speed up training but requires approximately 50GB of RAM.
- `--num_workers NUMBER` Defines the number of parallel threads that will load and preprocess inputs.
- `--use_mse True` If passed, will train the model using the mean squared error as loss function. If this is not desired, don't pass `False` but instead remove the argument entirely.

### Evaluate model

To evaluate a model, once training is finished, run the following line of code from the terminal.

```
python main.py evaluate --camels_root /path/to/CAMELS --run_dir path/to/model_run
```
This will calculate the discharge simulation for the validation period and store the results alongside the observed discharge for all basins in a pickle file. The pickle file is stored in the main directory of the model run.

## License of our code
[Apache License 2.0](https://github.com/kratzert/ealstm_regional_modeling/blob/master/LICENSE)

## License of the updated Maurer forcings and our pre-trained models
The CAMELS data set only allows non-commercial use. Thus, the updated Maurer forcings used underlie the same [TERMS OF USE](https://www2.ucar.edu/terms-of-use) as the CAMELS data set. 

## References:
[1] Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., & Nearing, G. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. Hydrology & Earth System Sciences, 23(12).  
[2] https://github.com/kratzert/ealstm_regional_modeling   
