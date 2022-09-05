# DeepVARwT: Deep Learning for a VAR Model with Trend
## Introduction
We propose a new approach called DeepVARwT that employs deep learning methodology for maximum likelihood estimation of the trend and the dependence structure at the same time. A Long Short-Term Memory (LSTM) network is used for this purpose. To ensure the stability of the model, we enforce the causality condition on the autoregressive coefficients using the transformation of Ansley & Kohn (1986). 

## Project structure
This repository contains python code and data used for reproducing results in simulation study and real data applications.

Here, we brifely introduce some important `.py` files in this project.

- `_main_for_para_estimation.py`: main code for parameter estimation in simulation study.
- `lstm_network.py`: set up an LSTM network to generate trend and VAR parameters.
- `custom_loss.py`: evaluate log-likelihood function.
- `_model_fitting_for_real_data.py`: model fitting for real data.
- `_main_make_predictions_for_real_data.py`: make predictions using fitted model.


## Preliminaries
All code was implemented using 
[![Python v3.6.15](https://img.shields.io/badge/python-v3.6.15-blue.svg)](https://www.python.org/downloads/release/python-3615/), and Pytorch was used for network training.

Installation in a virtual environment is recommended:
```
#install python with version 3.6.15
conda create --name python36 python=3.6.15
conda activate python36
#install pytorch with version 1.10.2
pip install torch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

The additonal installation of other packages with specific versions can be implemented using
```
pip install pandas==1.1.5 
pip install packaging==21.3 
pip install matplotlib==3.3.4
```
## Usage
#### Simulation study
The following command will do the parameter estimation on a simulated three-diemnsional VAR(2) procoess
```
python _main_for_para_estimation.py
```
The output of estimated VAR coefficient matrices and variance-covariance matrix of innovations is 
```
VAR coefficients
tensor([[-1.1027, -0.0938,  0.2779],
        [-0.6773, -0.4081, -0.1873],
        [ 0.2853,  0.4335,  0.3167]], grad_fn=<SelectBackward0>)
tensor([[-0.5458, -0.2654, -0.2364],
        [-0.4080,  0.4836,  0.3899],
        [-0.0205, -0.2652,  0.2650]], grad_fn=<SelectBackward0>)
Variance-covariance of innovations
tensor([[ 0.4280, -0.2636,  0.0963],
        [-0.2636,  0.3699,  0.0037],
        [ 0.0963,  0.0037,  0.4149]], grad_fn=<MmBackward0>)
```
The training loss values, estimated trends and pretrained-model file will be saved in the folder `simulation-res`.
#### Real data application
The following command will make predictions for the 20 training samples used in our working paper
```
_main_make_predictions_for_real_data.py
```
The output of forecasting accuracies in terms of mape and msis over h=1,...,8 is 
```
mape-ts1
[127.03773982 377.76535448 101.94123284 126.37218399 146.51827849 180.7473249  197.84411268 211.56709524]
mape-ts2
[36.55704384 48.42553785 49.16971866 50.02674209 41.00548526 43.16349429 42.14017047 36.36365334]
mape-ts3
[ 8.530108   11.81325803 18.03065788 28.70881071 40.22396588 51.20305959 62.72541781 77.85637804]
msis-ts1
[ 1.43821851  2.29101938  2.46945451  3.74550696  5.34402876  7.81317756 12.30963895 14.86744255]
msis-ts2
[ 8.91635074 10.61747258  8.4356214   6.22135386  7.09018367 11.09105811 8.08463584  7.09142066]
msis-ts3
[2.06339364 2.28184823 3.09500349 4.06178784 5.20472876 6.46300516 8.24345176 8.30124757]
```

References
----------

- Xixi Li, Jingsong Yuan (2022).  DeepVARwT: Deep Learning for a VAR Model with Trend.  [Working paper]().


