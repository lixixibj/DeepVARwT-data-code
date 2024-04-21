1  #!/usr/bin/env python36
2  # -*- coding: utf-8 -*-

import numpy as np
import torch
from custom_loss import *
from seed import *
import math
from lstm_network import DeepVARwT
from lstm_network import init_ar_param


def get_t_function_values_(seq_len,num_of_t):
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

    x = np.zeros(shape=(seq_len,num_of_t))
    t = (np.arange(seq_len) + 1) / seq_len

    for i in range(num_of_t):
        x[:,i]=t**(i+1)

    x = torch.from_numpy(x.reshape((seq_len,1,num_of_t)))

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
    all_stationary_coeffs = A_coeffs_for_causal_VAR(var_coeffs, order, m, var_cov_innovations_varp)
    for i in range(order):
        print(all_stationary_coeffs[:, :, i])



def train_network(data,num_of_t,num_layers,hidden_dim,iter1,iter2,m,order,lr,lr_trend,res_saving_path,threshould):
    r"""
        Network training.
        Parameters
        ----------
        train_data
           description: training data
           type: dataframe
           shape: (T,m)

        filtered_data
           description: filtered data from a two-sided filter for OLS in Phase1
           type: dataframe
           shape: (T-8*2,m+1)

        num_layers
           description: number of LSTM network layer
           type: int

        iter1
           description: number of iterations  for trend estimation in Phase 1
           type: int

        iter2
           description: number of iterations for trend  and AR parameter estimation in Phase 2
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
           description: the path for saving estimetd results
           type: str


        Returns
        -------
        pretrained_model
           description: pretrained model
    """
    

    seq_len = data.shape[0]
    m = data.shape[1]
    #get values of functions of t
    #x:shape(seq_len,batch,input_size)
    x=get_t_function_values_(seq_len,num_of_t)
    #y:shape(T,m)
    print('y-shape')
    y= torch.from_numpy(data.values)
    print(y.shape)

    lstm_model = DeepVARwT(input_size=x.shape[2],
                          hidden_dim=hidden_dim,
                          num_layers=num_layers,
                          seqence_len=x.shape[1],
                           m=m,
                           order=order)
    lstm_model = lstm_model.float()


    # # Update the parameter.
    # param.copy_(transformed_param)

    #weight_decay=1e-4
    optimizer = torch.optim.Adam(lstm_model.parameters(),lr=lr)
    log_likelihood=[]
    loss_trend=[]
    count_temp=1
    import os
    #create a folder to save estimated trends
    estimated_trend_file_path = res_saving_path+ 'trend/'
    trend_folder = os.path.exists(estimated_trend_file_path)
    if not trend_folder:  # 
        os.makedirs(estimated_trend_file_path)

    #create a folder to saving pretrained-model files
    pretrained_model_file_path = res_saving_path + 'pretrained_model/'
    pretrained_model_folder = os.path.exists(pretrained_model_file_path)
    if not pretrained_model_folder:  # 
        os.makedirs(pretrained_model_file_path)


#filtered data
    # df_filtered_ts=filtered_data.iloc[:,1:]
    # filtered_trend=torch.from_numpy(np.array(filtered_data.iloc[:,1:]))
    #begin to enter Phase 1 (initial trend estimation)
    for i in range(0,iter1):
        count_temp = 1 + count_temp
        var_coeffs, residual_parameters, trend = lstm_model(x.float())
        # trend_part=trend[8:158,:,:]
        # trend_error = compute_error_for_trend_estimation(target=filtered_trend.float(),
        #                                            trend=trend_part)
        #trend_part=trend[8:158,:,:]
        trend_error = compute_error_for_trend_estimation(target=y,
                                                   trend=trend)
        optimizer.zero_grad()
        trend_error.backward()
        optimizer.step()
        print('iterations' + str(i+1) + ':trend error')
        print(trend_error)
        loss_trend.append(trend_error.detach().numpy())
    #save trend estimation loss
    loss_df= pd.DataFrame({'trend_loss':loss_trend})
    loss_df.to_csv(estimated_trend_file_path+'trend_loss.csv')

    #begin to simultaneously estimate trend and AR parameter

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
         # trend forecast shape(T,m)
    de_trend_data=(y-trend[:,0,:]).detach().numpy()
    init_coeffs,L_elments=init_ar_param(de_trend_data,order,m)
    #init AR params using OLS
    state_dict = lstm_model.state_dict()
    
    for name, param in state_dict.items():# Transform the parameter as required.
        if name=='init_ar_parameters':
            param.copy_(init_coeffs)
        if name=='init_residual_params':
            param.copy_(L_elments)

    for i in range(iter2):
        var_coeffs, residual_parameters,trend= lstm_model(x.float())
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
    print_AR_params(var_coeffs, residual_parameters,m,order)
    #save pretrained-model
    model_name='iters_'+str(i) + '_t_'+str(num_of_t)+'_h_'+str(hidden_dim)+'_'+ 'pretrained_model.pkl'
    model_path = pretrained_model_file_path + model_name
    torch.save(lstm_model.state_dict(), model_path)
    
    return likelihood_loss,lstm_model,model_name

