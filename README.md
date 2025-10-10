# DeepVARwT: Deep Learning for a VAR Model with Trend (Li and Yuan, 2025)
## Introduction
We propose a new approach called DeepVARwT that employs deep learning methodology for maximum likelihood estimation of the trend and the dependence structure at the same time. A Long Short-Term Memory (LSTM) network is used for this purpose. To ensure the stability of the model, we enforce the causality condition on the autoregressive coefficients using the transformation of Ansley & Kohn (1986). 

Authors
-------

-   [Xixi Li](https://lixixibj.github.io/)
-   [Jingsong Yuan](https://www.research.manchester.ac.uk/portal/jingsong.yuan.html)

## Project structure
This repository contains python code and data used to reproduce results in a simulation study and real data applications.

Here, we brifely introduce some important `.py` files in this project.

- `_main_for_para_estimation_parallel_diff_seeds.py`: main code for 100 parameter estimation in a simulation study.
- `lstm_network.py`: set up an LSTM network to generate trend and VAR parameters.
- `custom_loss.py`: evaluate log-likelihood function.
- `_model_fitting_for_real_data.py`: model fitting for real data.
- `_main_make_predictions_for_real_data.py`: make predictions using the fitted model.

In addition, we have provided source code for reproducing forecasting results from DeepAR and DeepState in the folder `deepar-deepstate`.


## Preliminaries
All code was implemented using 
[![Python v3.6.15](https://img.shields.io/badge/python-v3.6.15-blue.svg)](https://www.python.org/downloads/release/python-3615/), and Pytorch was used for network training.

Installation in a virtual environment is recommended:
```
#install python with version 3.6.15
conda create --name mypy36 python=3.6.15
conda activate mypy36
pip install --upgrade typing-extensions
#install pytorch with version 1.10.2
pip install torch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 -f https://download.pytorch.org/whl/cpu/torch_stable.html

```

The additonal installation of other packages with specific versions can be implemented using
```
#for deepar and deepstate
pip install mxnet==1.7.0.post2 gluonts==0.8.1 numpy==1.19.5 pandas
pip install pandas==1.1.5 
pip install packaging==21.3 
pip install matplotlib==3.3.4
pip install statsmodels==0.12.2
pip install joblib
```
## Usage
#### Simulation study
The following code will do 100 parameter estimation on simulated three-diemnsional VAR(2) procoesses
```
python _main_for_para_estimation_parallel.py
```
The training loss function values, estimated trends and pretrained-model file will be saved in the folder `simulation-res/100-res/`.
#### US macroeconomics series 1
The following code will make predictions from 20 training samples
```
python _main_make_predictions_for_first_macro_data.py
```
The output of forecasting accuracies in terms of APE and SIS at h=1,...,8 is 
```
MAPE
[[684.79855215 860.7077644  153.96422119 162.42855077 164.44858316
  192.80701656 190.58221894 202.36723173]
 [ 66.63957118  67.87860229  56.24667681  57.42372743  65.04864481
   56.0903499   52.41970929  46.91254798]
 [  7.80038039  14.0074644   23.00198936  33.74414528  41.03356809
   52.67421515  66.82627425  80.52924528]]
MAPE h1-4
[465.47477213  62.04714443  19.63849486]
MAPE h1-8
[326.51301736  58.58247871  39.95216027]
MSIS
[[ 4.77524292  8.8643384  12.42108121 14.42593736 19.47687127 21.22492533
  25.2553493  29.09150483]
 [ 8.67073992  8.72883338  4.51829602  4.93369123  7.89813284  7.06888855
   4.57403183  4.55855382]
 [ 1.46817948  3.92015094  6.78510196  8.79043607 10.15164599 12.43657761
  15.2350458  17.30917874]]
MSIS h1-4
[10.12164997  6.71289014  5.24096711]
msis h1-8
[16.94190633  6.36889595  9.51203957]
num of parameters
[1350, 1350, 1350, 1230, 715, 1350, 1230, 1290, 1350, 1350, 1350, 1230, 715, 1290, 1290, 1230, 1350, 1350, 1230, 1350]
715
1350
```

#### Global temperatures
The following code will make predictions from 20 training samples
```
python _main_make_predictions_for_climate_p.py
```
The output of forecasting accuracies in terms of APE and SIS at h=1,...6 is 
```
MAPE
[[17.26119195 21.07883612 21.63234113 19.3799493  23.40652067 24.74336264]
 [22.91647407 25.13533074 21.03687782 25.00637265 32.09926447 34.66791184]
 [29.20457029 41.078845   38.77304777 28.77030446 30.89917632 32.94200466]]
MAPE h1-4
[19.99078973 23.02956088 36.35215435]
MAPE h1-8
[21.25036697 26.81037193 33.61132475]
MSIS
[[ 7.05453105 17.33643034 17.72616668 14.31955928 14.80240385 21.58639804]
 [ 6.47025725 11.08992829  7.54046606  6.57980372  9.2367093  11.64131129]
 [ 4.9959205  13.60231177 10.99670286  8.90232547  7.62092022 10.03155046]]
MSIS h1-4
[14.03904269  8.36688387  9.86497837]
msis h1-8
[15.47091487  8.75974598  9.35828854]
num of parameters
[508, 508, 444, 508, 476, 476, 444, 508, 508, 476, 444, 508, 508, 508, 508, 508, 508, 508, 508, 508]
444
508
```

#### US macroeconomics series 2
The following code will make predictions from 20 training samples
```
python_main_make_predictions_for_second_usmacro_data_bvarsv.py
```
The output of forecasting accuracies in terms of APE and SIS at h=1,...,8 is 
```
MAPE
[[13.60684539 23.72286253 29.44143896 34.29490993 32.91838924 33.62335993
  32.04137184 33.11062585]
 [ 3.26940604  5.5775901   7.54392872  9.88843303 11.98559665 14.18822139
  16.55756333 18.80520721]
 [ 7.16915314 11.1762116  11.12613994 12.52114752 13.12134075 15.80440503
  17.50018679 18.18972987]]
MAPE h1-4
[25.2665142   6.56983947 10.49816305]
MAPE h1-8
[29.09497546 10.97699331 13.32603933]
MSIS
[[ 2.00135522  4.47991412  6.24714163  8.94011157  8.94412825 10.15889653
  11.34286039 11.29511234]
 [ 1.24331201  2.41284325  3.35833852  5.46033362  6.58870619  8.79384743
  12.02568499 15.13660789]
 [ 1.40896523  2.67332776  2.60268694  3.07633527  3.54405734  4.60831061
   4.52653101  5.05364097]]
MSIS h1-4
[5.41713063 3.11870685 2.4403288 ]
msis h1-8
[7.92619    6.87745924 3.43673189]
num of parameters
[1290, 849, 1350, 1290, 1290, 1230, 1350, 1350, 1290, 1230, 1290, 1350, 1350, 897, 1350, 1350, 1290, 1350, 1230, 1350]
849
1350
```

References
----------

- Xixi Li, Jingsong Yuan (2022).  DeepVARwT: Deep Learning for a VAR Model with Trend.  [Journal of Applied Statistics](https://arxiv.org/abs/2209.10587).



