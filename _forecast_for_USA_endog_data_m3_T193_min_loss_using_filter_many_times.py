1  #!/usr/bin/env python36
2  # -*- coding: utf-8 -*-

import numpy as np
import torch
import random
from custom_loss import *
from seed import *
import random
seed_value=random.randint(2, 1000)
set_global_seed(seed_value)



def get_t_function_values_(series_len):
    r"""
        Get t function values.
        Parameters
        ----------
        series_len
           description: the length of ts
           type: str

        Returns
        -------
        time_functions_array
           description: array of time function values
           type: array
     """

    time_functions_array = np.zeros(shape=(3, series_len))
    t = (np.arange(series_len) + 1) / series_len
    time_functions_array[0, :] = t
    time_functions_array[1, :] = t * t
    time_functions_array[2, :] = t * t * t

    return time_functions_array



def get_data_and_time_function_values(data):
    r"""
        Prepare t function values as well as data for neural network training.
        Parameters
        ----------
        file_of_dataset
           description: the path of data
           type: str

        Returns
        -------
        train_data
           description: the observations of multivariate ts as well as time function values
           type: dict
    """
    seq_len=data.shape[0]
    train_data_and_t_function_values={}
    #time_feature_array: shape(covariates*seq_len)
    time_functions_array=get_t_function_values_(seq_len)
    time_functions_temp=time_functions_array

    time_functions_array1=time_functions_temp.transpose().tolist()
    time_functions=[]
    time_functions.append(time_functions_array1)
    time_func_array= np.array(time_functions)
    # original_shape: num.of.train.data*seq_t*num.of.features (batch,seq,input)
    # here we need to change the shape of the all_train_data_feature to(sep,batch,input)
    train_data_and_t_function_values['t_functions'] = torch.from_numpy(time_func_array)
    # shape: num.of.train.data*multivarite*seq_t
    # train_data['multi_target']=nd.array(np.array(all_train_data_target_list), dtype=nd.float64)
    #observations shape(seq_len*m)
    observations=data.iloc[:,1:]
    observations_array=np.array(observations)
    train_data_and_t_function_values['multi_target'] = torch.from_numpy(observations_array)

    return train_data_and_t_function_values





#original data: shape(batch,seq,input)
#changed data: shape(seq,batch,input)
def change_data_shape(original_data):
    #change to numpy from tensor
    original_data=original_data.numpy()
    new_data=[]
    for seq_temp in range(original_data.shape[1]):
        new_data.append(original_data[:,seq_temp,:].tolist())
    #change to tensor
    new_data=torch.from_numpy(np.array(new_data))
    return new_data



import math
from lstm_network import DeepVARwT
from lstm_network import DeepVARwT_biLSTM


def train_network(train_data,filtered_data,name_of_dataset,num_layers,hidden_dim,iterations_trend,iterations_AR,m,order,lr_trend,lr_ar,saving_path):
    #random.seed(0)
    #set_global_seed(2)
    data_and_t_function_values = get_data_and_time_function_values(train_data)
    #x:shape(seq*batch*input)
    x = data_and_t_function_values['t_functions']
    print('t_functions')
    print(x)
    sequence_length=x.shape[1]
    #y:shape(T*m)
    y = data_and_t_function_values['multi_target']
    print('target')
    print(y.shape)
    lstm_model = DeepVARwT(input_size=x.shape[2],
                          hidden_dim=hidden_dim,
                          num_layers=num_layers,
                          seqence_len=sequence_length,
                           m=m,
                           order=order)
    lstm_model = lstm_model.float()
    optimizer = torch.optim.Adam(lstm_model.parameters(),lr=lr_ar)
    likelihood=[]
    loss_trend=[]
    count_temp=1
    x_input = change_data_shape(x)
    import os
    #create folder for saving trend
    name_of_train_dataset = str.split(str.split(name_of_dataset, '/')[-1], '_')[0]
    path =saving_path +name_of_train_dataset+ '_h' + str(hidden_dim) + '_trend_iters_' + str(iterations_trend) + '_AR_iters_' + str(iterations_AR)+'_lr_t'+str(lr_trend)+'_lr_ar'+str(lr_ar)+'/'
    trend_file_path = path+ 'trend/'
    trend_folder = os.path.exists(trend_file_path)
    if not trend_folder:  # 
        os.makedirs(trend_file_path)

    # save pretrained-model
    pretrained_model_file_path = path + 'pretrained_model/'
    pretrained_model_folder = os.path.exists(pretrained_model_file_path)
    if not pretrained_model_folder:  # 
        os.makedirs(pretrained_model_file_path)

#prepare
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
    print('df_ts_shape')
    print(df_ts.shape)
    df_filtered_ts=filtered_data.iloc[:,1:]

    filtered_trend=torch.from_numpy(np.array(filtered_data.iloc[:,1:]))
    for iter in range(0,iterations_trend):
        count_temp = 1 + count_temp
        var_coeffs, redusial_parameters, trend_t = lstm_model(x_input.float())
        trend_t_part=trend_t[8:158,:,:]
        print('trend_t.shape')
        print(trend_t_part.shape)
        trend_error = compute_error_for_trend_estimation(target=filtered_trend.float(),
                                                   # trend shape(seq,batch,units(m))
                                                   trend_t=trend_t_part)
        optimizer.zero_grad()
        trend_error.backward()
        optimizer.step()
        print('iterations' + str(iter) + ':trend error')
        print(trend_error)
        #print(trend_error.detach().numpy())
        loss_trend.append(trend_error.detach().numpy())
        #save model

        trend_list = []
        for n in range(y.shape[1]):
            # one sub-seqence
            trend_flatten = torch.flatten(trend_t[:, :, n].t())
            trend_list.append(trend_flatten.tolist())
        df_trend = pd.DataFrame(np.transpose(np.array(trend_list)))
        #df_trend.to_csv(trend_file_path + str(iter) + '_.csv')
        #save pretrained model
        #model_name = pretrained_model_file_path + str(iter) + '_' + 'net_params.pkl'
        #torch.save(lstm_model.state_dict(), model_name)
        #plot ts vs trend
        if count_temp % 500== 0:
            # data frame
            import matplotlib.pyplot as plt
            if (len(df_ts.columns)) % 2 != 0:
                n_rows = int(len(df_ts.columns) / 2) + 1
            else:
                n_rows = int(len(df_ts.columns) / 2)
            fig, axes = plt.subplots(nrows=n_rows, ncols=2, dpi=150, figsize=(10, 10))
            print('nrows')
            print(int(len(df_ts.columns) / 2))
            print(df_ts.columns)
            for i, (col, ax) in enumerate(zip(df_ts.columns, axes.flatten())):
                df_ts.iloc[:, i].plot(legend=True, ax=ax, label='time series').autoscale(axis='x', tight=True)
                df_trend.iloc[:, i].plot(legend=True, ax=ax, label='trend');
                df_filtered_ts.iloc[:, i].plot(legend=True, ax=ax, label='filtered_data');  
                ax.set_title(str(col) + ": Time series, filtered data and trend")
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.spines["top"].set_alpha(0)
                ax.tick_params(labelsize=6)
            plt.tight_layout();
            #name_of_dataset1 = str.split(name_of_dataset, '/')[-1]
            name_of_dataset1 = str.split(str.split(name_of_dataset, '/')[-1], '_')[0]
            figure_name = name_of_dataset1 + '_' + str(x.shape[0]) + '_' + str(order) + '_' + str(count_temp) + '.png'
            fig.savefig(trend_file_path + figure_name)
            # fig.savefig('./image/simulation/p2_k3_all_penalty_epoch50000_h20_add_cov/trend/' + figure_name)
            plt.close()
            print_AR_params(lstm_model, x_input, y, m, order)
    #find min loss
    # index_of_model_for_trend=loss_trend.index(min(loss_trend))
    # print('index-of-model-for-trend')
    # print(index_of_model_for_trend)
    # pretrained_model_path = pretrained_model_file_path + str(index_of_model_for_trend) + '_' + 'net_params.pkl'
    # lstm_model.load_state_dict(torch.load(pretrained_model_path))
    print_AR_params(lstm_model, x_input, y, m, order)
    loss_pd= pd.DataFrame({'trend_loss':loss_trend})
    #folder_path='./image/simulation-trend-estimation-using-conditional-density-term-real-likelihood/p2_k3_all_penalty_epoch'+str(epoches)+'_h'+str(hidden_dim)+'_add_cov_update_stationarity_rep1/'

    loss_pd.to_csv(path+'trend_loss.csv')
    #  params dict_keys(['init_ar_parameters', 'init_redusial_params', 'lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0', 'add_trend.weight', 'add_trend.bias'])

    # optimizer = torch.optim.Adam([{'params': lstm_model.init_ar_parameters,'lr': 0.01},
    #     {'params': lstm_model.init_redusial_params, 'lr': 0.01}], lr=0.001)

    lstm_model.lstm.weight_ih_l0.requires_grad = True
    lstm_model.lstm.weight_hh_l0.requires_grad = True
    lstm_model.lstm.bias_ih_l0.requires_grad = True
    lstm_model.lstm.bias_hh_l0.requires_grad = True
    lstm_model.add_trend.weight.requires_grad = True
    lstm_model.add_trend.bias.requires_grad = True
    lstm_model.init_ar_parameters.requires_grad=True
    lstm_model.init_redusial_params.requires_grad=True

    optimizer = torch.optim.Adam([{'params': lstm_model.lstm.weight_ih_l0,'lr': lr_trend},
        {'params': lstm_model.lstm.weight_hh_l0, 'lr': lr_trend},{'params': lstm_model.lstm.bias_ih_l0,'lr': lr_trend},
        {'params': lstm_model.lstm.bias_hh_l0, 'lr': lr_trend},{'params': lstm_model.add_trend.weight,'lr': lr_trend},
        {'params': lstm_model.add_trend.bias, 'lr': lr_trend},{'params': lstm_model.init_ar_parameters,'lr': lr_ar},
        {'params': lstm_model.init_redusial_params, 'lr': lr_ar}], lr=0.01)


    for iter in range(iterations_trend,iterations_trend+iterations_AR):
        count_temp=1+count_temp
        var_coeffs, redusial_parameters,trend_t = lstm_model(x_input.float())
        likelihood_loss = compute_log_likelihood_using_conditional_density_term_no_penalty(target=y.float(),
                                                                            var_coeffs=var_coeffs.float(),
                                                                            redusial_parameters=redusial_parameters,
                                                                            m=m,
                                                                            order=order,
                                                                            trend_t=trend_t)
        optimizer.zero_grad()
        likelihood_loss.backward()
        optimizer.step()
        print('iterations'+str(iter)+':log-likelihood')
        print(likelihood_loss)
        likelihood.append(likelihood_loss.detach().numpy()[0,0])
        #save trend
        trend_list = []
        for n in range(y.shape[1]):
            # one sub-seqence
            trend_flatten = torch.flatten(trend_t[:, :, n].t())
            trend_list.append(trend_flatten.tolist())
        df_trend = pd.DataFrame(np.transpose(np.array(trend_list)))
        df_trend.to_csv(trend_file_path + str(iter) + '_.csv')
        #save pretrained model
        model_name = pretrained_model_file_path + str(iter) + '_' + 'net_params.pkl'
        torch.save(lstm_model.state_dict(), model_name)

        if count_temp%500==0:
            # x_orginal data: shape(batch,seq,input)
            import matplotlib.pyplot as plt
            if (len(df_ts.columns))%2!=0:
                n_rows=int(len(df_ts.columns) / 2)+1
            else:
                n_rows = int(len(df_ts.columns) / 2)
            fig, axes = plt.subplots(nrows=n_rows, ncols=2, dpi=150, figsize=(10, 10))
            # print('nrows')
            # print(int(len(df_ts.columns) / 2))
            # print(df_ts.columns)
            for i, (col, ax) in enumerate(zip(df_ts.columns, axes.flatten())):
                df_ts.iloc[:, i].plot(legend=True, ax=ax, label='time series').autoscale(axis='x', tight=True)
                df_trend.iloc[:, i].plot(legend=True, ax=ax, label='trend');
                ax.set_title(str(col) + ": Time series vs Trend")
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.spines["top"].set_alpha(0)
                ax.tick_params(labelsize=6)
            plt.tight_layout();

            # plt.show()
            import os
            name_of_dataset1 = str.split(name_of_dataset, '/')[-1]
            figure_name = name_of_dataset1+ '_'+str(x.shape[0]) + '_'+str(order) + '_'+str(count_temp)+'.png'
            fig.savefig(trend_file_path + figure_name)
            #fig.savefig('./image/simulation/p2_k3_all_penalty_epoch50000_h20_add_cov/trend/' + figure_name)
            plt.close()

            print_AR_params(lstm_model, x_input, y, m, order)

        #save loss
    index_of_model=likelihood.index(min(likelihood))+iterations_trend
    print('index-of-model')
    print(index_of_model)
    pretrained_model_path = pretrained_model_file_path + str(index_of_model) + '_' + 'net_params.pkl'
    lstm_model.load_state_dict(torch.load(pretrained_model_path))
    print_AR_params(lstm_model, x_input, y, m, order)
    loss_pd= pd.DataFrame({'likelihood':likelihood})
    #folder_path='./image/simulation-trend-estimation-using-conditional-density-term-real-likelihood/p2_k3_all_penalty_epoch'+str(epoches)+'_h'+str(hidden_dim)+'_add_cov_update_stationarity_rep1/'

    loss_pd.to_csv(path+'likelihood_loss.csv')




#print AR coeffs

def print_AR_params(lstm_model,x_input,y,m,order):
    var_coeffs, redusial_parameters, trend_t = lstm_model(x_input.float())
    likelihood_loss = compute_log_likelihood_using_conditional_density_term_no_penalty(target=y.float(),
                                                                                       var_coeffs=var_coeffs.float(),
                                                                                       redusial_parameters=redusial_parameters,
                                                                                       m=m,
                                                                                       order=order,
                                                                                       trend_t=trend_t)
    print('log-likelihood')
    print(likelihood_loss)

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


num_layers=1

iterations_trend=4000
iterations_AR=10000

name_of_dataset='/Users/xixili/Dropbox/deep-var-tend/deep-factors/data/endog_data_m3_T193.csv'
#filted_data_path='/Users/xixili/Dropbox/deep-var-tend/deep-factors/real-data/3_t_functions_order_2_reprdocue_T193/forecasts-residuals/filtered_data_T193_training_part.csv'
filted_data_path='/Users/xixili/Dropbox/deep-var-tend/deep-factors/benchmarks-R/filtered-data/'

saving_path='/Users/xixili/Dropbox/deep-var-tend/deep-factors/real-data/endog-forecast-many-times-p2-lrtrend001-same-seed-400/3-t-h20-5-groups/'+'seed'+str(seed_value)+'_m3_T193_using_3_t_'
m=3
order=2
lr_trend=0.001
lr_ar=0.01

train_data=pd.read_csv(name_of_dataset)

len_of_data=train_data.shape[0]
hidden_dim=20
num_layers=1
horizon=8
print('hidden-dim')
print(hidden_dim)
num_of_forecast=20
horizons=8
test_len=horizons+num_of_forecast-1

train_len=len_of_data-test_len

error=[]

nums=range(num_of_forecast)
nums=range(15,20)

nums=[5,12,17,18,19]
nums=[14,15]

for f in nums:
    b=f
    e=b+train_len
    training_data=train_data.iloc[b:e,:]
    filtered_data=pd.read_csv(filted_data_path+str(f+1)+'.csv').iloc[8:158,:]
    print('train_data_shape')
    print(training_data.shape)
    res_saving_path=saving_path+'section_'+str(f)
    try:
        train_network(training_data,filtered_data, name_of_dataset, num_layers, hidden_dim, iterations_trend, iterations_AR, m, order,lr_trend,lr_ar, res_saving_path)
    except:
        error.append(f)

print('error')
print(error)


