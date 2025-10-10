1  #!/usr/bin/env python36
2  # -*- coding: utf-8 -*-




import numpy as np
import torch
import random
from custom_loss import *
from seed import *
import math
from lstm_network import DeepVARwT
from lstm_network import init_ar_param
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
# seed_value=100
# set_global_seed(seed_value)


def get_t_function_values_(seq_len):
    r"""
        Get values of time functions.
        Parameters
        ----------
        seq_len
           description: the length of time series
           type: int
           shape: T

        Returns
        -------
        x
           description: tensor of time function values
           type: tensor
           shape: (seq_len,batch,input_size)
     """

    x = np.zeros(shape=(seq_len,6))
    t = (np.arange(seq_len) + 1) / seq_len
    x[:,0] = t
    x[:,1] = t * t
    x[:,2] = t * t * t
    inverse_t = 1 / (np.arange(seq_len) + 1)
    x[:,3] = inverse_t
    x[:,4] = inverse_t * inverse_t
    x[:,5] = inverse_t * inverse_t * inverse_t

    x = torch.from_numpy(x.reshape((seq_len,1,6)))

    return x



def plot_estimated_trend(m,df_trend,data,estimated_trend_file_path):
    r"""
        Plot estimated trends and observations.
        Parameters
        ----------
        df_trend
           description: estimated trends
           type: dataframe
           shape: (T,m)

        y
           description: observations
           type: dataframe
           shape: (T,m)  

        estimated_trend_file_path
           description: path for saving estimated trends
           type: str                 

        Returns
        -------
    """

    import matplotlib.pyplot as plt
    if (len(data.columns)) % 2 != 0:
        n_rows = int(len(data.columns) / 2) + 1
    else:
        n_rows = int(len(data.columns) / 2)
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, dpi=150, figsize=(10, 10))
    for i, (col, ax) in enumerate(zip(data.columns, axes.flatten())):
        data.iloc[:, i].plot(legend=True, ax=ax, label='time series').autoscale(axis='x', tight=True)
        df_trend.iloc[:, i].plot(legend=True, ax=ax, label='trend');
        ax.set_title(str(col) + ": Observations and Trend")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
    plt.tight_layout();
    fig.savefig(estimated_trend_file_path+'estimated_trend.png')
    plt.close()


def print_AR_params(var_coeffs, residual_parameters,m,order):

    r"""
        Print estimated AR parameters
        Parameters
        ----------
        var_coeffs
           description: VAR coefficients generated from LSTM
           type: tensor

        residual_parameters
           description: residual parameters generated from LSTM
           type: tensor   
           
        m
           description: number of time series
           type: int      

         order
           description: order of VAR model
           type: int                

        Returns
        -------
    """


    var_cov_innovations_varp = make_var_cov_matrix_for_innovation_of_varp(residual_parameters, m, order)
    all_coeffs = A_coeffs_for_causal_VAR(var_coeffs, order, m, var_cov_innovations_varp)
    for i in range(order):
        print(all_coeffs[:, :, i])
    print('var_cov_innovations_of_varp')
    print(var_cov_innovations_varp)
    return all_coeffs,var_cov_innovations_varp



def train_network(data,num_layers,hidden_dim,iter1,iter2,lr,lr_trend,m,order,res_saving_path,threshould):
    r"""
        Network training.
        Parameters
        ----------
        data
           description: training data
           type: dataframe
           shape: (T,m)

        num_layers
           description: number of LSTM network layer
           type: int

        hidden_dim
           description: number of units of LSTM network
           type: int

        iter1
           description: number of iterations  for trend estimation in Phase 1
           type: int

        iter2
           description: number of iterations for trend and AR parameter estimation in Phase 2
           type: int

        lr
           description: learning rate for trend estimation in Phase 1  and for AR parameter estimation in Phase 2
           type: int      

        lr_trend
           description: learning rate in Phase 2 for trend estimation
           type: int

        m
           description: number of time series
           type: int      

         order
           description: order of VAR model
           type: int           

        res_saving_path
           description: the path for saving estimated results
           type: str


        Returns
        -------
    """
    seq_len = data.shape[0]
    m = data.shape[1]
    #get values of functions of t
    #x:shape(seq_len,batch,input_size)
    x=get_t_function_values_(seq_len)
    #y:shape(T,m)
    y= torch.from_numpy(data.values)
    lstm_model = DeepVARwT(input_size=x.shape[2],
                          hidden_dim=hidden_dim,
                          num_layers=num_layers,
                          seqence_len=x.shape[1],
                           m=m,
                           order=order)
    lstm_model = lstm_model.float()
    optimizer = torch.optim.Adam(lstm_model.parameters(),lr=lr)
    log_likelihood=[]
    loss_trend=[]
    import os
    #create a folder to save estimated trends
    estimated_trend_file_path = res_saving_path+ 'trend/'
    trend_folder = os.path.exists(estimated_trend_file_path)
    if not trend_folder:  # 
        os.makedirs(estimated_trend_file_path)

    #create a folder to save pretrained-model files
    pretrained_model_file_path = res_saving_path + 'pretrained_model/'
    pretrained_model_folder = os.path.exists(pretrained_model_file_path)
    if not pretrained_model_folder:  # 
        os.makedirs(pretrained_model_file_path)


#begin to enter Phase 1 (initial trend estimation)
    for i in range(0,iter1):
        var_coeffs, residual_parameters, trend = lstm_model(x.float())
        #OLS estimation for trend
        trend_error = compute_error_for_trend_estimation(target=y.float(),
                                                   # trend shape(seq,batch,m)
                                                   trend=trend)
        optimizer.zero_grad()
        trend_error.backward()
        optimizer.step()
        print('iterations' + str(i) + ':trend error')
        print(trend_error)
        #print(trend_error.detach().numpy())
        loss_trend.append(trend_error.detach().numpy())
    loss_df= pd.DataFrame({'trend_loss':loss_trend})
    loss_df.to_csv(estimated_trend_file_path+'trend_loss_phase1.csv')

    #begin to simultaneously estimate trend and AR parameter
      #begin to enter Phase 2
         # trend forecast shape(T,m)
    de_trend_data=(y-trend[:,0,:]).detach().numpy()
    init_coeffs,L_elments=init_ar_param(de_trend_data,order,m)
    #init AR params using OLS
    state_dict = lstm_model.state_dict()
    
    for name, param in state_dict.items():# Transform the parameter as required.
        if name=='init_ar_parameters':
            print('param')
            print(param)
            print(init_coeffs)
            param.copy_(init_coeffs)
        if name=='init_residual_params':
            param.copy_(L_elments)

    #setting gradient of parameters as true means that all these parameters will be used for the computation of log-likelihood in Phase 2
    #LSTM parameters for trend
    lstm_model.lstm.weight_ih_l0.requires_grad = True
    lstm_model.lstm.weight_hh_l0.requires_grad = True
    lstm_model.lstm.bias_ih_l0.requires_grad = True
    lstm_model.lstm.bias_hh_l0.requires_grad = True
    lstm_model.add_trend.weight.requires_grad = True
    lstm_model.add_trend.bias.requires_grad = True
    #AR parameters
    lstm_model.init_ar_parameters.requires_grad=True
    lstm_model.init_residual_params.requires_grad=True

    #set different learning rates for different parameters
    optimizer = torch.optim.Adam([{'params': lstm_model.lstm.weight_ih_l0,'lr': lr_trend},
        {'params': lstm_model.lstm.weight_hh_l0, 'lr': lr_trend},{'params': lstm_model.lstm.bias_ih_l0,'lr': lr_trend},
        {'params': lstm_model.lstm.bias_hh_l0, 'lr': lr_trend},{'params': lstm_model.add_trend.weight,'lr': lr_trend},
        {'params': lstm_model.add_trend.bias, 'lr': lr_trend},{'params': lstm_model.init_ar_parameters,'lr': lr},
        {'params': lstm_model.init_residual_params, 'lr': lr}])

#begin to enter Phase 2
    total_params = sum(p.numel() for p in lstm_model.parameters())
    print(f" total params: {total_params}")
    for i in range(iter2):
        var_coeffs, residual_parameters,trend = lstm_model(x.float())
        likelihood_loss = compute_log_likelihood(target=y.float(),
                                                 var_coeffs=var_coeffs.float(),
                                                 residual_parameters=residual_parameters,
                                                 m=m,
                                                 order=order,
                                                 trend=trend)
        optimizer.zero_grad()
        likelihood_loss.backward()
        optimizer.step()
        print('iterations'+str(i+1)+':log-likelihood')
        print(likelihood_loss)
        #print AR coefficients
        all_coeffs,var_cov_innovations_varp=print_AR_params(var_coeffs, residual_parameters,m,order)
        log_likelihood.append(likelihood_loss.detach().numpy()[0,0])
        if i>=2:
            current_loss=log_likelihood[i]
            past1_loss=log_likelihood[i-1]
            abs_relative_error1=abs((current_loss-past1_loss)/past1_loss)
            if abs_relative_error1<threshould:
                past1_loss=log_likelihood[i-1]
                past2_loss=log_likelihood[i-2]
                abs_relative_error2=abs((past1_loss-past2_loss)/past2_loss)
                if abs_relative_error2<threshould:
                    #end loop
                    break

    #save loss values
    loss_df= pd.DataFrame({'likelihood':log_likelihood})
    loss_df.to_csv(res_saving_path+'log_likelihood_loss_phase2.csv')  
    #saving estimated trend
    trend_list = []
    for n in range(m):
        trend_flatten = torch.flatten(trend[:, :, n].t())
        trend_list.append(trend_flatten.tolist())
    df_trend = pd.DataFrame(np.transpose(np.array(trend_list))) 
    df_trend.to_csv(estimated_trend_file_path+'estimated_trend.csv')  
    #plot estimated trend
    plot_estimated_trend(m,df_trend,data,estimated_trend_file_path)
    #print AR coefficients
    all_coeffs,var_cov_innovations_varp=print_AR_params(var_coeffs, residual_parameters,m,order)
    #save pretrained-model
    model_name = pretrained_model_file_path + str(i) + '_' + 'pretrained_model.pkl'
    torch.save(lstm_model.state_dict(), model_name)

    return all_coeffs,var_cov_innovations_varp



def process_section(num,data_saving_path,num_layers,hidden_dim, iter1,iter2,lr,lr_trend, m, order,res_saving_path,threshould):
    path_of_dataset=data_saving_path+'exp'+str(num)+'_len800_VAR2_m3_train.csv'
    data=pd.read_csv(path_of_dataset).iloc[:, 1:]
    res_saving_path1=res_saving_path+str(num)+'/'
    try:
        set_global_seed(100)
        all_coeffs,var_cov_m=train_network(data, num_layers,hidden_dim, iter1,iter2,lr,lr_trend, m, order,res_saving_path1,threshould)
    except:
        set_global_seed(200)
        all_coeffs,var_cov_m=train_network(data, num_layers,hidden_dim, iter1,iter2,lr,lr_trend, m, order,res_saving_path1,threshould)
        #error.append(i)
        print('error')
        print(num)
    return all_coeffs,var_cov_m




############train network#################
import timeit
start = timeit.default_timer()
m=3
order=2
lr_trend=0.001
lr=0.01
hidden_dim=20
num_layers=1
iter1=13000
iter2=600
threshould=1e-5

data_saving_path='./simulation-study/simulated_data_len800_VAR2_m3/'
res_saving_path='./simulation-study/100-res-011025-rep/'
#100 estimation
exps=100
# Run the process in parallel
results = Parallel(n_jobs=-1)(delayed(process_section)(num, data_saving_path,num_layers,hidden_dim, iter1,iter2,lr,lr_trend, m, order,res_saving_path,threshould) for num in range(exps))

# Extract results

coeffs_all=np.zeros((exps,m*m*order))
var_cov_all=np.zeros((exps,m*m))
for num, (all_coeffs,var_cov_m) in enumerate(results):
    coffes=[]
    for i in range(order):
        coffes.extend(all_coeffs[:, :, i].detach().numpy().flatten().tolist())
    coeffs_all[num,:]=coffes
    var_cov_all[num,:]=var_cov_m.detach().numpy().flatten().tolist()

#save 
pd.DataFrame(coeffs_all).to_csv('./simulation-study/100-res-011025-rep/all_coeffs_100exps_011025_rep.csv')
pd.DataFrame(var_cov_all).to_csv('./simulation-study/100-res-011025-rep/all_var_cov_100exps_011025_rep.csv')



stop = timeit.default_timer()


print('Time: ', stop - start) 



