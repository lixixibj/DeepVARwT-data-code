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
#### Real data application 1
The following code will make predictions from 20 training samples
```
python _main_make_predictions_for_real_data.py
```
The output of forecasting accuracies in terms of APE and SIS at h=1,...,8 is 
```
MAPE
[[671.56949393 877.26671665 154.23600535 162.32857398 164.32171888
  192.61076685 190.67055743 202.38820978]
 [ 66.75856615  67.94758772  56.26046135  57.35064394  64.91021096
   56.0912243   52.38016094  46.97814736]
 [  7.83372128  14.03229333  23.09566731  33.6761764   40.90670453
   52.56566649  66.77089378  80.4494223 ]]
MAPE h1-4
[466.35019748  62.07931479  19.65946458]
MAPE h1-8
[326.92400536  58.58462534  39.91631818]
MSIS
[[ 4.84710644  8.96068356 12.48652455 14.50909102 19.52114304 21.26635881
  25.31208447 29.15764974]
 [ 8.7113116   8.72724434  4.45512754  4.91191965  7.87588789  7.06629155
   4.52470348  4.5580891 ]
 [ 1.4729179   3.94621703  6.82237222  8.80507067 10.16005861 12.4659966
  15.27679226 17.38148287]]
MSIS h1-4
[10.20085139  6.70140078  5.26164445]
msis h1-8
[17.0075802   6.35382189  9.54136352]
num of parameters
[1350, 1350, 1350, 1230, 715, 1350, 1230, 1290, 1350, 1350, 1350, 1230, 715, 1290, 1290, 1230, 1350, 1350, 1230, 1350]
715
1350
Time:  3344.0242955870926
```

References
----------

- Xixi Li, Jingsong Yuan (2022).  DeepVARwT: Deep Learning for a VAR Model with Trend.  [Working paper](https://arxiv.org/abs/2209.10587).



