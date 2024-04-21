1  #!/usr/bin/env python36
2  # -*- coding: utf-8 -*-




import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

import pandas

import statsmodels.api as sm

from statsmodels.tsa.api import VAR


def init_ar_param(de_trend_data,p,m):
    #print('testttttttt')
    #data: shape(T,m), np.array
    #https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.var_model.VAR.fit.html#statsmodels.tsa.vector_ar.var_model.VAR.fit
    model = VAR(de_trend_data)
    results = model.fit(p,method='ols',trend='n')
    para=results.params
    print(para)
    ncol=para.shape
    all_p = torch.randn(m, m, p)
    #count=1
    for i in range(p):
        all_p[:,:,i]=torch.t(torch.from_numpy(np.array(para[i*m:(i*m+m),:])))

    # for i in range(p):
    #     for j in range(m):
    #         all_p[:,j,i]=torch.from_numpy(np.array(para[count,:]))
    #         count=count+1
    L=np.linalg.cholesky(np.matrix(results.sigma_u))

    L_elems=[]
    count_temp=0
    #make lower triangel matrix
    for i in range(m):
        for j in range(i+1):
            L_elems.append(L[i,j])
            count_temp=count_temp+1

    return all_p, torch.tensor(L_elems)


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
        #self.lstm=nn.GRU(self.input_size,self.hidden_dim,self.num_layers)
        #self.lstm=nn.RNN(self.input_size,self.hidden_dim,self.num_layers)
        #init AR coefficients
        num_of_ar_coefficients=self.order*self.m*self.m
        #self.init_ar_parameters = torch.nn.Parameter(torch.randn(int(num_of_ar_coefficients)))
        self.init_ar_parameters = torch.nn.Parameter(torch.randn(m, m, order))
        num_of_residual_parmas=(self.m*(self.m+1))/2
        #init residual parameters (lower triangular matrix)
        self.init_residual_params=torch.nn.Parameter(torch.randn(int(num_of_residual_parmas)))
        self.add_trend = nn.Linear(self.hidden_dim, self.m)


    def forward(self,inputs):
        output, (hn, cn) = self.lstm(inputs)
        #output, hn = self.lstm(inputs)
        residual_parameters=self.init_residual_params
        trend=self.add_trend(output)
        var_coeffs=self.init_ar_parameters
        return var_coeffs, residual_parameters,trend







