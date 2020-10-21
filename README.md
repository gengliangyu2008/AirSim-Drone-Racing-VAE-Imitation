# AirSim-Drone-Racing-VAE-Imitation

This repository forked from below repo, please use original repo for details:<br/>
https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation 

## Purpose of this repo
Changes here aim to make it to work well in windows env with python3.
Also to make it able to be run easily in Pycharm.

## Python Environment
**Notice**<br/>
Current project is using tensorflow-2.0.0b1<br/>
For linux, tensorflow-2 is able to run in python2<br/> 
For windows, tensorflow-2 is only allowed in python3<br/> 

Linux: please refer to file "VAE-PY2-ENV.yml" in root folder<br/> 
Windows without GPU: please refer to file "WINDOWS-VAE-PY3-ENV.yml" in root folder<br/> 
Windows with GPU: please refer to file "WINDOWS-VAE-PY3-ENV-V2.yml" in root folder<br/> 

## How to enable GPU

Please refer to nvidia_gpu_enable_steps.txt

## Files which have been run successfully so far: 
Need to replace **base_dir** to your own folder where data file was placed<br/>
1. train_cmvae.py
2. eval_cmvae.py
3. train_bc.py
4. bc_navigation.py - airsim env needs to be run first

## Steps
1. Open the root folder with pycharm directly.
2. select the according env created with above yml.
3. replace the path in train_cmvae.py and train_cmvae.py, then will be able to run these 2.

## Pre Conditions
1. Data files need to be downloaded and unzipped.
2. Check whether the path match with what has been specified in py codes