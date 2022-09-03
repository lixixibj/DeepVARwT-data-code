# DeepVARwT: Deep Learning for a VAR Model with Trend
## Introduction
We propose a new approach called DeepVARwT that employs deep learning methodology for maximum likelihood estimation of the trend and the dependence structure at the same time. A Long Short-Term Memory (LSTM) network is used for this purpose. To ensure the stability of the model, we enforce the causality condition on the autoregressive coefficients using the transformation of \cite{ansley1986note}. 

## Preliminaries
All code was implemented using: 
[![Python v3.6.15](https://img.shields.io/badge/python-v3.6.15-blue.svg)](https://www.python.org/downloads/release/python-3615/).
Pytorch was used for network training.

Installation in a virtual environment is recommended:
```
#install python with version 3.6.15
conda create --name python36 python=3.6.15
conda activate python36
#install pytorch with version 1.10.2
pip install torch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

The additonal installation of other packages with specific versions can be performed using
```
pip install pandas==1.1.5 
pip install packaging==21.3 
pip install matplotlib==3.3.4
```
## Usage
# Simulation study
The following command will do parameter estimation on a simulated VAR(2) procoess
```
python _main_for_para_estimation.py
```
The output is 
```
hhhhh
```
The training loss values, estimated trends and pretrained-model will be saved in the folder `simulation-res`.
# Real data application
The following command will make predictions for 20 training samples used in our working paper
```
_main_make_predictions_for_real_data.py
```

