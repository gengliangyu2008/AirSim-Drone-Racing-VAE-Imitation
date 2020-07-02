# AirSim-Drone-Racing-VAE-Imitation

This repository forked from below repo, please use original repo for details:<br/>
https://github.com/microsoft/AirSim-Drone-Racing-VAE-Imitation 

## Purpose of this repo
Changes here aim to make it to work well in windows env with python3.
Also to make it able to be run easily in Pycharm.

## Environment
Please refer to file "environment.yml" in root folder

## Files which have been run successfully so far: 
1. train_cmvae.py, need to replace C:/Users/gary/Downloads/ to your own folde where data file was placed.
2. eval_cmvae.py, need to replace C:/Users/gary/Downloads/ to your own folde where data file was placed.

## Steps
1. Open the root folder with pycharm directly.
2. select the according env created with above yml.
3. replace the path in train_cmvae.py and train_cmvae.py, then will be able to run these 2.

## Pre Conditions
1. Data files need to be downloaded and unzipped.
2. Check whether the path match with what has been specified in py codes