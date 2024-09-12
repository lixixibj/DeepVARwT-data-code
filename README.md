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

- `_main_for_para_estimation_parallel.py`: main code for 100 parameter estimation in a simulation study.
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
pip install --upgrade typing-extensions
#install pytorch with version 1.10.2
pip install torch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 -f https://download.pytorch.org/whl/cpu/torch_stable.html

```

The additonal installation of other packages with specific versions can be implemented using
```
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
MSE
[[0.90668418 1.93563334 2.74771814 3.2049125  4.33622751 4.86934014
  5.90989885 6.66686321]
 [1.49309915 1.50409208 1.14383033 1.06739008 1.51637498 1.41770861
  1.13532586 0.98392444]
 [0.29328373 0.86875047 1.79275641 2.29163343 2.69861961 3.51833753
  4.39531109 4.9434979 ]]
MSE h1-4
[2.19873704 1.30210291 1.31160601]
MSE h1-8
[3.82215973 1.28271819 2.60027377]
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
```

#### Global temperatures
The following code will make predictions from 20 training samples
```
python _main_make_predictions_for_climate_p.py
```
The output of forecasting accuracies in terms of APE and SIS at h=1,...6 is 
```
MSE
[[0.01684853 0.03933144 0.04247139 0.03546891 0.04343313 0.05747433]
 [0.00949617 0.01475883 0.0104524  0.01189918 0.01645997 0.01807153]
 [0.01848354 0.04190784 0.03850571 0.03075166 0.03169977 0.03816988]]
MSE h1-4
[0.03288379 0.01156913 0.0329657 ]
MSE h1-8
[0.03917129 0.01352301 0.03325307]
MAPE
[[16.85218399 21.447882   21.33873643 19.16176024 22.86178841 25.36380648]
 [23.28367484 24.09010244 20.71467121 23.02103492 30.06367694 31.32879839]
 [30.20710811 41.15049183 38.67960263 29.75650979 31.62711035 34.30813719]]
MAPE h1-4
[19.87960081 22.6961495  36.67906752]
MAPE h1-8
[21.17102626 25.41699312 34.28815998]
MSIS
[[ 6.95513903 17.07918004 17.9982927  14.37725997 15.21246431 22.31044879]
 [ 6.51698169 11.10123283  7.58434565  6.84212592  8.25638845  8.49366746]
 [ 4.98581308 13.59242729 11.26477168  8.95860554  8.17819301 10.05791533]]
MSIS h1-4
[14.01087059  8.40085339  9.94767068]
msis h1-8
[15.65546414  8.132457    9.50628765]
num of parameters
[508, 508, 444, 508, 476, 476, 444, 508, 508, 476, 444, 444, 444, 508, 508, 508, 508, 508, 508, 508]
444
508
Time:  2031.6952918219613
```

#### US macroeconomics series 2
The following code will make predictions from 20 training samples
```
python_main_make_predictions_for_second_usmacro_data_bvarsv.py
```
The output of forecasting accuracies in terms of APE and SIS at h=1,...,8 is 
```
MAPE
[[13.70702837 24.01910302 30.06146535 34.79659754 33.5687801  34.40913268
  33.1306328  34.28307669]
 [ 3.44682309  5.84526161  8.02545794  9.83299817 11.72115349 13.70107581
  15.41108278 17.26570431]
 [ 7.02431293 11.04490492 10.76419691 11.68813492 12.78012413 14.79760403
  16.21274527 16.69894343]]
MAPE h1-4
[25.64604857  6.78763521 10.13038742]
MAPE h1-8
[29.74697707 10.65619465 12.62637082]
MSIS
[[ 1.88236843  4.62555044  6.55369797  9.15022675  9.08398762 10.17772461
  11.01439385 10.91750569]
 [ 1.2394466   2.23888376  3.12877997  5.17721549  6.2015562   7.93419333
  10.42537603 13.05860833]
 [ 1.408648    2.72788475  2.59454041  3.04464974  3.53034703  4.48494289
   4.03598523  4.04754621]]
MSIS h1-4
[5.5529609  2.94608146 2.44393072]
msis h1-8
[7.92568192 6.17550746 3.23431803]
num of parameters
[1290, 849, 1350, 1290, 1290, 1230, 1350, 1350, 1290, 1230, 1290, 1350, 1350, 897, 1350, 1350, 1290, 1350, 1230, 1350]
849
1350
Time:  3542.271742865909
```

References
----------

- Xixi Li, Jingsong Yuan (2022).  DeepVARwT: Deep Learning for a VAR Model with Trend.  [Working paper](https://arxiv.org/abs/2209.10587).



