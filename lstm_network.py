1  #!/usr/bin/env python35
2  # -*- coding: utf-8 -*-
3  # @File  : lstm_network.py
4  # @Author: Xixi Li
5  # @Date  : 2019-12-02
6  # @Desc  :



import numpy as np
import torch
import torch.nn as nn
#from utils import ParameterBounds

# rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)#(input_size,hidden_size,num_layers)
# input = torch.randn(5, 3, 10)#(seq_len, batch, input_size)
# h0 = torch.randn(2, 3, 20) #(num_layers,batch,output_size)
# c0 = torch.randn(2, 3, 20) #(num_layers,batch,output_size)
# output, (hn, cn) = rnn(input, (h0, c0))

import torch.nn.functional as F



#定义网络结构
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
        #initial AR coefficients
        num_of_ar_coefficients=self.order*self.m*self.m
        self.init_ar_parameters = torch.nn.Parameter(torch.randn(int(num_of_ar_coefficients)))
        #import torch.nn.functional as F
        #prior covariance shape:(batch,(m*m+1/2))
        num_of_redusial_parmas=(self.m*(self.m+1))/2
        #Q covariance shape(seq_len,batch,(m*m+1/2))
        #self.Q_t_model=nn.Linear(1,int(parmas))
        self.init_redusial_params=torch.nn.Parameter(torch.randn(int(num_of_redusial_parmas)))
        # 2020-03-24 add trend shape(seq,batch,units)
        #add another linear
        self.add_trend = nn.Linear(self.hidden_dim, self.m)


    def forward(self,inputs):

        #features
        #output.shape #(seq_len, batch, output_size=hidden_size)
        output, (hn, cn) = self.lstm(inputs)
        # output, (hn, cn) = self.lstm(inputs)
        #generate A coefficients

        #inputs_A=torch.zeros(1, 1)
        #shape:(1*num_A)
        #F_coeffs=self.F_t_model(inputs_A)
        #generate elements for Q(var-covar)
        #inputs_Q= torch.zeros(1, 1)
        # shape:(1*num_A)
        #Q_coeffs=self.Q_t_model(inputs_Q)
        #Q_coeffs:shape(n)
        redusial_parameters=self.init_redusial_params
        #Q_coeffs = lower_and_upper(Q_coeffs, (-1, 0), (0, 1))
        #generate trend
        trend_t=self.add_trend(output)
        #trend_t=postive_interval(trend_t, (1,5))
        #shape(n)
        var_coeffs=self.init_ar_parameters

        #return F_coeffs,Q_coeffs,prior_mean,prior_cov_input
        return var_coeffs, redusial_parameters,trend_t



class DeepVARwT_biLSTM(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        super(DeepVARwT_biLSTM,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m=m
        self.order=order
        self.lstm=nn.LSTM(self.input_size,self.hidden_dim,self.num_layers,bidirectional=True)
        #initial AR coefficients
        num_of_ar_coefficients=self.order*self.m*self.m
        self.init_ar_parameters = torch.nn.Parameter(torch.randn(int(num_of_ar_coefficients)))
        #import torch.nn.functional as F
        #prior covariance shape:(batch,(m*m+1/2))
        num_of_redusial_parmas=(self.m*(self.m+1))/2
        #Q covariance shape(seq_len,batch,(m*m+1/2))
        #self.Q_t_model=nn.Linear(1,int(parmas))
        self.init_redusial_params=torch.nn.Parameter(torch.randn(int(num_of_redusial_parmas)))
        # 2020-03-24 add trend shape(seq,batch,units)
        #add another linear
        self.add_trend = nn.Linear(self.hidden_dim*2, self.m)


    def forward(self,inputs):

        #features
        #output.shape #(seq_len, batch, output_size=hidden_size)
        output, (hn, cn) = self.lstm(inputs)
        # output, (hn, cn) = self.lstm(inputs)
        #generate A coefficients

        #inputs_A=torch.zeros(1, 1)
        #shape:(1*num_A)
        #F_coeffs=self.F_t_model(inputs_A)
        #generate elements for Q(var-covar)
        #inputs_Q= torch.zeros(1, 1)
        # shape:(1*num_A)
        #Q_coeffs=self.Q_t_model(inputs_Q)
        #Q_coeffs:shape(n)
        redusial_parameters=self.init_redusial_params
        #Q_coeffs = lower_and_upper(Q_coeffs, (-1, 0), (0, 1))
        #generate trend
        trend_t=self.add_trend(output)
        #trend_t=postive_interval(trend_t, (1,5))
        #shape(n)
        var_coeffs=self.init_ar_parameters

        #return F_coeffs,Q_coeffs,prior_mean,prior_cov_input
        return var_coeffs, redusial_parameters,trend_t


#定义网络结构
class DeepVARwT_3layers(torch.nn.Module):
    def __init__(self,input_size,hidden_dim,num_layers,seqence_len,m,order):
        super(DeepVARwT_3layers,self).__init__()
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.seqence_len=seqence_len
        self.m=m
        self.order=order
        self.lstm=nn.LSTM(self.input_size,5,self.num_layers)
        #initial AR coefficients
        num_of_ar_coefficients=self.order*self.m*self.m
        self.init_ar_parameters = torch.nn.Parameter(torch.randn(int(num_of_ar_coefficients)))
        #import torch.nn.functional as F
        #prior covariance shape:(batch,(m*m+1/2))
        num_of_redusial_parmas=(self.m*(self.m+1))/2
        #Q covariance shape(seq_len,batch,(m*m+1/2))
        #self.Q_t_model=nn.Linear(1,int(parmas))
        self.init_redusial_params=torch.nn.Parameter(torch.randn(int(num_of_redusial_parmas)))
        # 2020-03-24 add trend shape(seq,batch,units)
        #add another linear
        self.add_linear = nn.Linear(5, self.hidden_dim)
        self.add_trend = nn.Linear(self.hidden_dim, self.m)


    def forward(self,inputs):

        #features
        #output.shape #(seq_len, batch, output_size=hidden_size)
        output, (hn, cn) = self.lstm(inputs)
        # output, (hn, cn) = self.lstm(inputs)
        #generate A coefficients

        #inputs_A=torch.zeros(1, 1)
        #shape:(1*num_A)
        #F_coeffs=self.F_t_model(inputs_A)
        #generate elements for Q(var-covar)
        #inputs_Q= torch.zeros(1, 1)
        # shape:(1*num_A)
        #Q_coeffs=self.Q_t_model(inputs_Q)
        #Q_coeffs:shape(n)
        redusial_parameters=self.init_redusial_params
        #Q_coeffs = lower_and_upper(Q_coeffs, (-1, 0), (0, 1))
        #generate trend
        output=self.add_linear(output)
        trend_t=self.add_trend(output)
        #trend_t=postive_interval(trend_t, (1,5))
        #shape(n)
        var_coeffs=self.init_ar_parameters

        #return F_coeffs,Q_coeffs,prior_mean,prior_cov_input
        return var_coeffs, redusial_parameters,trend_t




import torch
from torch.autograd import Variable
def transform_data_size(inital_data,m,lag_order):
    #change to numpy
    #1.#(seq_len, batch, output_size=hidden_size) ------->batch,seq,output
    inital_data=inital_data.data.numpy()
    all_data=[]
    for b_temp in range(inital_data.shape[1]):
        all_data.append(inital_data[:,b_temp,:].tolist())
    all_data=np.array(all_data)
    #2.batch,seq,output----------->batch*seqence_len*(F_t)
    all_list=[]
    for batch in range(inital_data.shape[0]):
        b_lis = []
        for seq in range(inital_data.shape[1]):
            units_elm = inital_data[batch, seq,]
            np_val = units_elm.reshape(m, m*lag_order)
            # add zero and dig matrix
            dig_array = np.diag(np.array([0] * (m*lag_order-m)))
            zero_array = np.diag(np.array([0] * m))
            added = np.concatenate((dig_array, zero_array), axis=1)
            final = list(np.concatenate((np_val, added), axis=0))
            b_lis.append(final)
        #
        all_list.append(b_lis)
    #
    transformed_data = np.array(all_list)

    return Variable(torch.from_numpy(transformed_data))
# print(transform_data_size(output_new,4,2).shape)




# Q_coeffs = Q_coeffs * (self.prior_cov_bounds.upper - self.prior_cov_bounds.lower) \
#                   + self.prior_cov_bounds.lower

#Q_coeffs = (1/(1+torch.exp(-Q_coeffs))) * (self.prior_cov_bounds.upper - self.prior_cov_bounds.lower) \
#                          + self.prior_cov_bounds.lower
#original_data: shape(batch,units)
#neg_interval: range(-0.03,0) dict
#positive_interval: range(0,0.03)
def lower_and_upper(original_data,neg_interval,post_interval):
    print('test')
    print(original_data.shape)
    for i in range(original_data.shape[0]):
        for j in range(original_data.shape[1]):
            if original_data[i,j]<0:
                original_data[i, j]=(1/(1+torch.exp(-original_data[i, j]))) * (neg_interval[1]-neg_interval[0])+neg_interval[0]
            else:
                original_data[i, j] = (1 / (1 + torch.exp(-original_data[i, j]))) * (post_interval[1] - post_interval[0]) + post_interval[0]
    return (original_data)

#min-max interval
def min_max_lower_and_upper(original_data,min_max_range):
    print('test')
    print(original_data.shape)
    for i in range(original_data.shape[0]):
        #min and max
        min_value=min(original_data[i,:])
        max_value=max(original_data[i,:])
        for j in range(original_data.shape[1]):
            original_data[i, j]=((original_data[i, j]-min_value) * (min_max_range[1]-min_max_range[0]))/(max_value-min_value)+min_max_range[0]
    return (original_data)


#trend original shape:test torch.Size([100, 60, 8])(seq*batch*m)
def trend_interval(original_data):
    #interval=[(0.4, 1.1), (1.3, 2.2), (0.6, 1.1), (0.1, 0.23), (0.1, 0.22), (0, 0.014), (0.3, 0.9), (0.5, 0.84)]
    interval = [(-2, 2), (-2, 2), (-2, 2),(-2, 2)]
    for ts_num in range(original_data.shape[2]):
        ts=original_data[:,:,ts_num]
        original_data[:, :, ts_num]=postive_interval(ts,interval[ts_num])
    return original_data


def postive_interval(original_data,post_interval):
    print('test')
    print(original_data.shape)
    for i in range(original_data.shape[0]):
        for j in range(original_data.shape[1]):
            original_data[i, j] = (1 / (1 + torch.exp(-original_data[i, j]))) * (post_interval[1] - post_interval[0]) + post_interval[0]
    return (original_data)





