# Intelligent-Navigation-Systems

## Project structure introduction(key packages and folders)

1. cmvae package, train and evaluation CMVAE python codes were included here.
2. imitation_learning package, includes train behavior cloning and navigation python files.
3. racing_models package, all the tensorflow modes were included in this package.
4. model_outputs folder, includes all the trained models and output logs accordingly.
5. tello package, include API classes which convert airsim commands to tello commands, tello control UI and video codes.


## Software Environment

Ubuntu 18.04 with Python 2.7.15 or<br/>
Windows 10 with Python 3.7.7<br/>
As the airsim env is based on linux env, so window env can only be used for model training and testing purpose.

**Notice**<br/>
Current project is using tensorflow-2.0.0b1<br/>
For linux, tensorflow-2 is able to run in python2<br/> 
For windows, tensorflow-2 is only allowed in python3<br/> 

Linux: please refer to file "VAE-PY2-ENV.yml" in root folder<br/> 
Windows without GPU: please refer to file "WINDOWS-VAE-PY3-ENV.yml" in root folder<br/> 
Windows with GPU: please refer to file "WINDOWS-VAE-PY3-ENV-V2.yml" in root folder<br/>

## Dataset and airsim virtual environment files

https://drive.google.com/drive/folders/19tFUG8bCg3_d_PeQMDHJQvj-ZBv8Ogs_

## How to enable GPU for model training

Please refer to nvidia_gpu_enable_steps.txt in documents folder