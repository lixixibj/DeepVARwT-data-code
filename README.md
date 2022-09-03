# DeepVARwT: Deep Learning for a VAR Model with Trend
## Introduction
We propose a new approach called DeepVARwT that employs deep learning methodology for maximum likelihood estimation of the trend and the dependence structure at the same time. A Long Short-Term Memory (LSTM) network is used for this purpose. To ensure the stability of the model, we enforce the causality condition on the autoregressive coefficients using the transformation of \cite{ansley1986note}. 
## Project structure

### Preliminaries
All code was implemented using: 
[![Python v3.6.15](https://img.shields.io/badge/python-v3.6.15-blue.svg)](https://www.python.org/downloads/release/python-3615/)
Pytorch was used for network training.

See `requirements.txt` for package versions - installation in a virtual environment is recommended:
```
conda create --name env python=3.8
conda activate env
pip install -r requirements.txt
```
When training on GPU machines, the appropriate PyTorch bundle should be installed - for more info: `https://pytorch.org/get-started/locally/` 

Note the additonal required install in the requirements which can be performed with:
```
pip install git+https://github.com/openai/CLIP.git
```
