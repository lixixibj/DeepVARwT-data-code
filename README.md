# DeepVARwT: Deep Learning for a VAR Model with Trend (Li and Yuan, 2022)
## Introduction
We propose a new approach called DeepVARwT that employs deep learning methodology for maximum likelihood estimation of the trend and the dependence structure at the same time. A Long Short-Term Memory (LSTM) network is used for this purpose. To ensure the stability of the model, we enforce the causality condition on the autoregressive coefficients using the transformation of Ansley & Kohn (1986). 

Authors
-------

-   [Xixi Li](https://lixixibj.github.io/)
-   [Jingsong Yuan](https://www.research.manchester.ac.uk/portal/jingsong.yuan.html)

## Project structure
This repository contains python code and data used to reproduce results in a simulation study and real data applications.

Here, we brifely introduce some important `.py` files in this project.

- `_main_for_para_estimation.py`: main code for parameter estimation in simulation study.
- `lstm_network.py`: set up an LSTM network to generate trend and VAR parameters.
- `custom_loss.py`: evaluate log-likelihood function.
- `_model_fitting_for_real_data.py`: model fitting for real data.
- `_main_make_predictions_for_real_data.py`: make predictions using the fitted model.


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
The following code will do parameter estimation on a simulated three-diemnsional VAR(2) procoess
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
The training loss function values, estimated trends and pretrained-model file will be saved in the folder `simulation-res`.
#### Real data application
The following code will make predictions from 20 training samples
```
python _main_make_predictions_for_real_data.py
```
The output of forecasting accuracies in terms of APE and SIS at h=1,...,8 is 
```
MAPE
[[173.5362957  304.16919248  74.21356812 115.58417515 130.6897122
  165.08193326 160.87637934 173.92743636]
 [ 47.10828891  49.51429175  46.0766192   46.93643932  66.51316581
   53.95455726  54.67764628  43.65015813]
 [  6.26311105   9.4015523   17.11915371  27.27283024  37.0598462
   48.88284836  67.33157402  78.62430559]]
MSIS
[[ 1.31672002  2.09134377  2.88309118  3.59235625  6.41856765  9.55948713
  11.29671458 14.91358247]
 [13.5144462   9.24089923  6.32883524  6.2892517  12.32530752 12.60838306
   7.7972476   7.37495114]
 [ 1.95195755  2.3432803   2.79117611  4.04551831  4.50571452  5.91031199
  10.10513132 11.43978964]]
```

References
----------

- Xixi Li, Jingsong Yuan (2022).  DeepVARwT: Deep Learning for a VAR Model with Trend.  [Working paper](https://arxiv.org/abs/2209.10587).



