1  #!/usr/bin/env python36
2  # -*- coding: utf-8 -*-




import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F




#define network structure
class DeepVARwT(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        super(DeepVARwT,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m=m
        self.order=order
        self.lstm=nn.LSTM(self.input_size,self.hidden_dim,self.num_layers)
        #init AR coefficients
        num_of_ar_coefficients=self.order*self.m*self.m
        self.init_ar_parameters = torch.nn.Parameter(torch.randn(int(num_of_ar_coefficients)))
        num_of_residual_parmas=(self.m*(self.m+1))/2
        #init residual parameters (lower triangular matrix)
        self.init_residual_params=torch.nn.Parameter(torch.randn(int(num_of_residual_parmas)))
        self.add_trend = nn.Linear(self.hidden_dim, self.m)


    def forward(self,inputs):
        output, (hn, cn) = self.lstm(inputs)
        residual_parameters=self.init_residual_params
        trend=self.add_trend(output)
        var_coeffs=self.init_ar_parameters
        return var_coeffs, residual_parameters,trend







