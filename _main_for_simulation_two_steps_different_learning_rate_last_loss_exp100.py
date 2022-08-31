1  #!/usr/bin/env python36
2  # -*- coding: utf-8 -*-




import numpy as np
import torch
import random
from custom_loss import *
from seed import *
import math
from lstm_network import DeepVARwT
seed_value=100
set_global_seed(seed_value)


def get_t_function_values_(sample_size):
    r"""
        Get values of t functions.
        Parameters
        ----------
        sample_size
           description: the length of time series
           type: int
           shape: T

        Returns
        -------
        time_functions_array
           description: array of time function values
           type: array
           shape: (6,T)
     """

    time_functions_array = np.zeros(shape=(6, sample_size))
    t = (np.arange(sample_size) + 1) / sample_size
    time_functions_array[0, :] = t
    time_functions_array[1, :] = t * t
    time_functions_array[2, :] = t * t * t
    inverse_t = 1 / (np.arange(sample_size) + 1)
    time_functions_array[3, :] = inverse_t
    time_functions_array[4, :] = inverse_t * inverse_t
    time_functions_array[5, :] = inverse_t * inverse_t * inverse_t

    return time_functions_array



def get_data_and_time_function_values(path_of_dataset):
    r"""
        Prepare t function values and data for neural network training.
        Parameters
        ----------
        path_of_dataset
           description: the path of data storage
           type: str

        Returns
        -------
        data and time function values
           description: the observations of multivariate time series and values of time functions
           type: dict
    """
    data = pd.read_csv(path_of_dataset)
    sample_size = data.shape[0]
    data_and_t_function_values = {}
    # time_function_values_array: shape(6*seq_len)
    time_functions_array = get_t_function_values_(sample_size)
    # observations shape(sample_size*m)
    observations = data.iloc[:, 1:]
    # observations: shape(m*sample_size)
    observations = np.array(observations).T
    time_functions_temp = time_functions_array
    time_functions_array1 = time_functions_temp.transpose().tolist()
    time_functions = []
    time_functions.append(time_functions_array1)
    t_func_array = np.array(time_functions)
    # the original shape of time functions array: (seq=sample_size,input_size=6)
    # here we need to change the shape of which to(sep=sample_size,batch=1,input_size=6)
    data_and_t_function_values['t_functions'] = torch.from_numpy(t_func_array)
    observations = data.iloc[:, 1:]
    observations_array = np.array(observations)
    # observations_array: shape(sample_size,m)
    data_and_t_function_values['multi_target'] = torch.from_numpy(observations_array)

    return data_and_t_function_values



def change_data_shape(original_data):
    r"""
        Change shape of data.
        Parameters
        ----------
        original_data
           description: the original data
           type: tensor
           shape: (batch,seq,input)

        Returns
        -------
        transformed data 
           description: transformed data
           type: tensor
           shape: (seq,batch,input)
    """
    #change to numpy from tensor
    original_data=original_data.numpy()
    new_data=[]
    for seq_temp in range(original_data.shape[1]):
        new_data.append(original_data[:,seq_temp,:].tolist())
    #change to tensor
    new_data=torch.from_numpy(np.array(new_data))
    return new_data


def plot_estimated_trend(m,df_trend,df_ts,estimated_trend_file_path):


    import matplotlib.pyplot as plt
    if (len(df_ts.columns)) % 2 != 0:
        n_rows = int(len(df_ts.columns) / 2) + 1
     else:
        n_rows = int(len(df_ts.columns) / 2)
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, dpi=150, figsize=(10, 10))
    for i, (col, ax) in enumerate(zip(df_ts.columns, axes.flatten())):
        df_ts.iloc[:, i].plot(legend=True, ax=ax, label='time series').autoscale(axis='x', tight=True)
        df_trend.iloc[:, i].plot(legend=True, ax=ax, label='trend');
        ax.set_title(str(col) + ": Time series vs Trend")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
    plt.tight_layout();
    fig.savefig(estimated_trend_file_path+'estimated_trend.png')
    plt.close()



def train_network(path_of_dataset,num_layers,hidden_dim,iterations_trend,iterations_AR,lr,lr_trend,m,order,res_saving_path):
    r"""
        Network training.
        Parameters
        ----------
        path_of_dataset
           description: the path of data storage
           type: str

        num_layers
           description: number of LSTM network layer
           type: int

        iterations_trend
           description: number of iterations  for trend estimation in Phase 1
           type: int

        iterations_AR
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
        data and time function values
           description: the observations of multivariate time series and values of time functions
           type: dict
    """

    data_and_t_function_values = get_data_and_time_function_values(path_of_dataset)
    #x:shape(seq=sample_size,batch=1,input_size=6)
    x = data_and_t_function_values['t_functions']
    # sequence_length=x.shape[1]
    #y:shape(T*m)
    y = data_and_t_function_values['multi_target']
    lstm_model = DeepVARwT(input_size=x.shape[2],
                          hidden_dim=hidden_dim,
                          num_layers=num_layers,
                          seqence_len=x.shape[1],
                           m=m,
                           order=order)
    lstm_model = lstm_model.float()
    optimizer = torch.optim.Adam(lstm_model.parameters(),lr=lr)
    likelihood=[]
    loss_trend=[]
    count_temp=1
    #x_input: shape: (seq_len=sample_size,batch=1,input_size=6)
    x_input = change_data_shape(x)
    import os
    #create folder for saving estimated trend
    estimated_trend_file_path = res_saving_path+ 'trend/'
    trend_folder = os.path.exists(estimated_trend_file_path)
    if not trend_folder:  # 
        os.makedirs(estimated_trend_file_path)

    # save pretrained-model
    pretrained_model_file_path = res_saving_path + 'pretrained_model/'
    pretrained_model_folder = os.path.exists(pretrained_model_file_path)
    if not pretrained_model_folder:  # 
        os.makedirs(pretrained_model_file_path)

#prepare data for plotting estiamted trend and observations
    ts_list = []
    for n in range(y.shape[1]):
        # one sub-seqence
        ts = y[:, n]
        ts_flatten = torch.flatten(ts)
        ts_list.append(ts_flatten.tolist())
    # change to dataframe
    import pandas as pd
    # df_ts=pd.DataFrame(np.transpose(np.array(ts_list))).iloc[0:1800,:]
    df_ts = pd.DataFrame(np.transpose(np.array(ts_list)))


    for iter in range(0,iterations_trend):
        count_temp = 1 + count_temp
        var_coeffs, redusial_parameters, trend = lstm_model(x_input.float())
        #OLS estimation for trend
        trend_error = compute_error_for_trend_estimation(target=y.float(),
                                                   # trend shape(seq,batch,units(m))
                                                   trend=trend)
        optimizer.zero_grad()
        trend_error.backward()
        optimizer.step()
        print('iterations' + str(iter) + ':trend error')
        print(trend_error)
        #print(trend_error.detach().numpy())
        loss_trend.append(trend_error.detach().numpy())
    loss_pd= pd.DataFrame({'trend_loss':loss_trend})
    loss_pd.to_csv(estimated_trend_file_path+'trend_loss.csv')

    #begin to simultaneously train trend and AR parameter

    #setting gradient of parameters as true means that all these parameters to be used for the computation of log-likelihood in Phase 2
    #LSTM parameters for trend
    lstm_model.lstm.weight_ih_l0.requires_grad = True
    lstm_model.lstm.weight_hh_l0.requires_grad = True
    lstm_model.lstm.bias_ih_l0.requires_grad = True
    lstm_model.lstm.bias_hh_l0.requires_grad = True
    lstm_model.add_trend.weight.requires_grad = True
    lstm_model.add_trend.bias.requires_grad = True
    #AR parameters
    lstm_model.init_ar_parameters.requires_grad=True
    lstm_model.init_redusial_params.requires_grad=True

    #set different learning rates
    optimizer = torch.optim.Adam([{'params': lstm_model.lstm.weight_ih_l0,'lr': lr_trend},
        {'params': lstm_model.lstm.weight_hh_l0, 'lr': lr_trend},{'params': lstm_model.lstm.bias_ih_l0,'lr': lr_trend},
        {'params': lstm_model.lstm.bias_hh_l0, 'lr': lr_trend},{'params': lstm_model.add_trend.weight,'lr': lr_trend},
        {'params': lstm_model.add_trend.bias, 'lr': lr_trend},{'params': lstm_model.init_ar_parameters,'lr': lr},
        {'params': lstm_model.init_redusial_params, 'lr': lr}])


    for i in range(iterations_AR):

        count_temp=1+count_temp
        var_coeffs, redusial_parameters,trend = lstm_model(x_input.float())
        likelihood_loss = compute_log_likelihood(target=y.float(),
                                                 var_coeffs=var_coeffs.float(),
                                                 redusial_parameters=redusial_parameters,
                                                 m=m,
                                                 order=order,
                                                 trend_t=trend)
        optimizer.zero_grad()
        likelihood_loss.backward()
        optimizer.step()
        print('iterations'+str(iter+1)+':log-likelihood')
        print(likelihood_loss)
        likelihood.append(likelihood_loss.detach().numpy()[0,0])
        if i>=2:
            current_loss=likelihood[i]
            past1_loss=loss_df.iloc[i-1]
            abs_relative_error1=abs((current_loss-past1_loss)/past1_loss)
            if abs_relative_error1<threshould:
                past1_loss=loss_df.iloc[i-1]
                past2_loss=loss_df.iloc[i-2]
                abs_relative_error2=abs((past1_loss-past2_loss)/past2_loss)
                if abs_relative_error2<threshould:
                    #end loop
                    break

    #save loss values
    loss_pd= pd.DataFrame({'likelihood':likelihood})
    loss_pd.to_csv(res_saving_path+'likelihood_loss.csv')  
    #saving estimated trend
    trend_list = []
    for n in range(m):
        trend_flatten = torch.flatten(trend[:, :, n].t())
        trend_list.append(trend_flatten.tolist())
    df_trend = pd.DataFrame(np.transpose(np.array(trend_list))) 
    df_trend.to_csv(estimated_trend_file_path+'likelihood_loss.csv')  
    #plot estimated trend
    plot_estimated_trend(m,df_trend,df_ts,estimated_trend_file_path)
    #print AR coefficients
    print_AR_params(var_coeffs, redusial_parameters, trend,m,order)


 


#print AR coeffs

def print_AR_params(var_coeffs, redusial_parameters, trend,m,order):


    # all_stationary_coeffs = staionary_coefs(initial_var_coeffs=var_coeffs, p=lag_order, d=m)
    var_cov_innovations_varp = make_var_cov_matrix_for_innovation_of_varp(redusial_parameters, m, order)
    all_stationary_coeffs = A_coeffs_for_causal_VAR(var_coeffs, order, m, var_cov_innovations_varp)
    # all_A.shape:d*d*p
    print(all_stationary_coeffs.shape)
    # print('F_sub')
    print('Coefficients of VAR')
    for i in range(order):
        print(all_stationary_coeffs[:, :, i])
    # print ('lower)
    print('Elements of lower matrix')
    print(redusial_parameters)
    print('var_cov_innovations_varp')
    print(var_cov_innovations_varp)







############begin to train and test network#################

m=3
order=2
lr_trend=0.001
lr=0.01





hidden_dim=45
hidden_dim=40
hidden_dim=35

hidden_dim=20


num_layers=1
iterations_trend=15000
iterations_AR=6000

data_saving_path='/Users/xixili/Dropbox/deep-var-tend/deep-factors/sim_exp100/simulated_data_len800_VAR2_m3_train.csv/'
#saving_path='./image/simulation-estimation-res-two-steps-diff-lr/p2_k3'
res_saving_path='/Users/xixili/Dropbox/deep-var-tend/deep-factors/sim_exp100/res-len800-var2-m3/'
print('hidden-dim')
print(hidden_dim)
ranges_list=range(170,175)
error=[]
for i in ranges_list:
    res_saving_path_1=res_saving_path+str(seed_value)+'_seed_'
    name_of_dataset=data_saving_path+'exp'+str(i)+'_len800_VAR2_m3'
    try:
        train_network(name_of_dataset, num_layers,hidden_dim, iterations_trend,iterations_AR,lr,lr_trend, m, order,res_saving_path_1)
    except:
        error.append(i)
print('error')
print(error)
