1  #!/usr/bin/env python36
2  # -*- coding: utf-8 -*-


from custom_loss import *
import numpy as np
import pandas as pd
import torch
import os
from forecasting_accuracy import *
from lstm_network import *
from _model_fitting_for_real_data import *




def get_time_function_values(len_of_train_and_test,horizon):
    r"""
        Get values of time functions.
        Parameters
        ----------
        len_of_train_and_test
           description: the length of training and test part of time series
           type: int
           length: T+horizon

        horizon
           description: forecasting horizon
           type: int
           length: h

        Returns
        -------
        time_functions_array
           description: array of time function values
           type: array
           shape: (3,len_of_train_and_test)
     """

    len_of_train_ts=len_of_train_and_test-horizon
    time_feature_array = np.zeros(shape=(3, len_of_train_and_test))
    for i in range(len_of_train_and_test):
        t=(i+1)/len_of_train_ts
        time_feature_array[0,i]=t
        time_feature_array[1, i] =t*t
        time_feature_array[2, i] = t * t*t

    return time_feature_array




def get_data_and_time_function_values(train_test_data,horizon):
    r"""
        Prepare time function values and data.
        Parameters
        ----------
        train_test_data
           description: training and test data
           type: dataframe
           shape: (T+h,m+1)

        Returns
        -------
        data_and_t_function_values
           description: the observations of time series and values of time functions
           type: dict
    """

    data_and_time_func_values = {}
    data=train_test_data
    seq_len=data.shape[0]
    time_feature_array=get_time_function_values(seq_len,horizon)
    observations=data.iloc[:,1:]
    observations=np.array(observations).T
    time_feature_temp=time_feature_array
    time_feature_array1=time_feature_temp.transpose().tolist()
    time_features=[]
    time_features.append(time_feature_array1)
    time_func_array= np.array(time_features)
    data_and_time_func_values['t_functions'] = torch.from_numpy(time_func_array)
    observations=data.iloc[:,1:]
    observations_array=np.array(observations)
    data_and_time_func_values['multi_target'] = torch.from_numpy(observations_array)

    return data_and_time_func_values






def change_data_shape(original_data):
    r"""
        Change shape of data.
        Parameters
        ----------
        original_data
           description: the original data
           type: tensor
           shape: (batch,seq,input_size)

        Returns
        -------
        transformed data 
           description: transformed data
           type: tensor
           shape: (seq,batch,input_size)
    """
    #change to numpy from tensor
    original_data=original_data.numpy()
    new_data=[]
    for seq_temp in range(original_data.shape[1]):
        new_data.append(original_data[:,seq_temp,:].tolist())
    #change to tensor
    new_data=torch.from_numpy(np.array(new_data))
    return new_data








def forecast_based_on_pretrained_model(train_test_data,m,order,pretrained_model,horizon,seasonality):
    r"""
        Network training.
        Parameters
        ----------
        train_test_data
           description: training and test data
           type: dataframe
           shape: (T+h,m+1)

         order
           description: order of VAR model
           type: int           

        m
           description: number of series
           type: int


        pretrained_model
           description: pretrained model

        horizon
           description: forecasting horizon
           type: int      


        seasonality
           description: seasonality of time series
           type: int      


        Returns
        -------
        mse_list
           description: mse value over different horizons
           type: list
           length: h

        mape_list
           description: mape value over different horizons
           type: list
           length: h

        msis_list
           description: msis value over different horizons
           type: list
           length: h

        point_forecast_array
           description: point forecasts
           type: array
           shape: (m,h)

        final_lower_forecast_array
           description: lower forecasts
           type: array
           shape: (m,h)

        final_upper_forecast_array
           description: upper forecasts
           type: array
           shape: (m,h)

    """


    data_and_time_func_values = get_data_and_time_function_values(train_test_data,horizon)
    x = data_and_time_func_values['t_functions']
    y = data_and_time_func_values['multi_target']
    lstm_model = pretrained_model
    train_test_len = x.shape[1]
    train_len=train_test_len-horizon
    observations = np.array(train_test_data.iloc[:, 1:])
    x_input = change_data_shape(x)
    var_coeffs, residual_parameters, trend= lstm_model(x_input.float())
    #begin to forecast
    # trend forecast shape(h,m)------>(m,h)
    trend_forecast=trend[train_len:,0,:].t()
    #last_p_trend shape(p,m)
    last_p_trend=trend[train_len-order:train_len,0,:]
    #lagged p observations
    p_lagged_value = []
    for i in range(order):
        obs = observations[(train_len - 1 - i), :].reshape(m, 1)
        detrend_obs=torch.from_numpy(obs).float()-last_p_trend[(order-1-i),:].reshape(m,1)
        p_lagged_value.append(detrend_obs)
    p_lagged_observations = torch.cat(p_lagged_value, dim=0)
    mp = m * order


    var_cov_innovations_varp = make_var_cov_matrix_for_innovation_of_varp(residual_parameters, m, order)
    all_causal_coeffs = A_coeffs_for_causal_VAR(var_coeffs, order, m, var_cov_innovations_varp)
    A_coeffs_var1 = get_A_coeff_m_for_VAR_1(all_causal_coeffs, m, order)

    A_list = []
    for c in range(order):
        A_list.append(all_causal_coeffs[:, :, c])
    A_all = torch.cat(A_list, dim=1)


    forecasts_list=[]
    upper_forecast_list = []
    lower_forecast_list = []

    identy_m=torch.eye(m)
    zeros_cols = torch.zeros([m, (mp - m)])
    #J: shape(k,kp)
    J= torch.cat((identy_m, zeros_cols), 1)

    for t in range(horizon):
            #get A coeffis
            #forecasts:shape:(m,1)
        A_multipication=multipy_A_matrix(A_coeffs_var1,(t+1),mp)
        forecasts=torch.mm(A_multipication,p_lagged_observations)
        #take part forecasts from forecasts
        forecasts_part=torch.mm(J,forecasts)
        forecasts_list.append(forecasts_part)
            #cal interval prediction
        square_root_list=cal_var_cov_of_prediction_error(A_coeffs_var1,residual_parameters,(t+1),order,m)
        #2 make interval prediction
        upper_forecast=forecasts_part+torch.from_numpy(np.array(square_root_list).reshape(m,1)*1.96)
        lower_forecast = forecasts_part -torch.from_numpy(np.array(square_root_list).reshape(m, 1) * 1.96)
        upper_forecast_list.append(upper_forecast)
        lower_forecast_list.append(lower_forecast)


    forecasts_ar=torch.cat(forecasts_list, dim=1)
            # print(forecasts_ar.shape)
    upper_forecast_ar = torch.cat(upper_forecast_list, dim=1)
    lower_forecast_ar = torch.cat(lower_forecast_list, dim=1)
    print('ar-forecasts')
    print(forecasts_ar.shape)
        # #trend+ar
    print('trend_forecast')
    print(trend_forecast.shape)
    final_forecast=forecasts_ar+trend_forecast
    final_upper_forecast = upper_forecast_ar + trend_forecast
    final_lower_forecast = lower_forecast_ar + trend_forecast
    print('ar forecast')
    print((forecasts_ar.detach().numpy()))
        #cal accuracy
    point_forecast_array=final_forecast.detach().numpy()
    final_upper_forecast_array = final_upper_forecast.detach().numpy()
    final_lower_forecast_array = final_lower_forecast.detach().numpy()
        #acutal_observations:shape(horizon,m)
    acutal_observations=observations[train_len:,:]
    print('actual foreecat')
    print((acutal_observations))
    print('final forecast')
    print(point_forecast_array)
    mse_list = []
    mape_list = []
    msis_list = []


    for i in range(m):
        print('ts')
        print(i)
        print('actucal-forecast')
        print(acutal_observations[:, i])
        print('mse')
        print(mse_cal(acutal_observations[:, i], point_forecast_array[i, :]))
        mse_list.append(list(mse_cal(acutal_observations[:, i], point_forecast_array[i, :])))
        print('mape')
        print(mape_cal(acutal_observations[:,i], point_forecast_array[i,:]))
        mape_list.append(mape_cal(acutal_observations[:,i], point_forecast_array[i,:]))
        print('msis')
        print(msis(observations[0:train_len,i], acutal_observations[:,i],
                       final_upper_forecast_array[i,:], final_lower_forecast_array[i,:], 0.05, seasonality, horizon))
        msis_list.append(msis(observations[0:train_len,i], acutal_observations[:,i],
                       final_upper_forecast_array[i,:], final_lower_forecast_array[i,:], 0.05, seasonality, horizon))

    return mse_list,mape_list,msis_list,point_forecast_array,final_lower_forecast_array,final_upper_forecast_array









def cal_var_cov_of_prediction_error(A,residual_parameters,horizon,order,m):
    r"""
        Calculate variance-covariance matrix of prediction error of different horizons
        Parameters
        ----------
        A
           description: coefficient matrix of VAR(1)
           type: tensor
           shape: (m*p,m*p)

        residual_parameters
           description: residual parameters
           type: tensor
           shape: (m*(m+1)/2,)

        horizon
           description: forecasting horizon
           type: int  

        order
           description: order of VAR model
           type: int           

        m
           description: number of series
           type: int


        Returns
        -------
        sd_list 
           description: sd of prediction error
           type: list
           length: m
    """

    mp=m*order
    identy_m=torch.eye(m)
    zeros_cols = torch.zeros([m, (mp - m)])
    #J: shape(k,kp)
    J= torch.cat((identy_m, zeros_cols), 1)

    # U_covariance:shape(mp*mp)
    U_covariance = make_var_covar_matrix(residual_parameters, m, order)

    var_cov_temp=torch.zeros(mp,mp)
    for i in range(horizon):
        A_multip=multipy_A_matrix(A,i,mp)
        var_cov_temp=var_cov_temp+torch.mm(torch.mm(A_multip,U_covariance),A_multip.t())
    var_cov=torch.mm(torch.mm(J,var_cov_temp),J.t())
    #cal h_step forecast variance of each time series
    sd_list=[]
    import math
    for n in range(m):
        sd_list.append(math.sqrt(var_cov[n,n]))
    return sd_list


#calculate FF^{i} power law of matrix
def multipy_A_matrix(FF,i,mp):
    A_multiply_tmp = torch.eye(mp, mp)
    if i==0:
        return torch.eye(mp, mp)
    else:
        for num in range(i):
            A_multiply_tmp=torch.mm(A_multiply_tmp,FF)
        return A_multiply_tmp





############begin to forecasting#################
m=3
order=2
lr_trend=0.001
lr=0.01
seasonality=4
name_of_dataset='./real-data/endog_data_m3_T193.csv'
filted_data_path='./real-data/filtered-data/'
train_data=pd.read_csv(name_of_dataset)

len_of_data=train_data.shape[0]
hidden_dim=20
num_layers=1
iterations_trend=4000
iterations_AR=9700
# iterations_trend=100
# iterations_AR=50
num_of_forecast=20
horizons=8
test_len=horizons+num_of_forecast-1
train_len=len_of_data-test_len
threshould=0.000002

saving_path='./real-data-forecasting-res/'
#seed values for reproducting forecasting results
seed_value_list=[400,400,400,400,400,
                 400,400,400,400,400,
                 400,400,600,400,400,
                 400,600,600,100,300]

nums=range(num_of_forecast)
all_mape_ts1=np.zeros((num_of_forecast,horizons))
all_msis_ts1=np.zeros((num_of_forecast,horizons))
all_mape_ts2=np.zeros((num_of_forecast,horizons))
all_msis_ts2=np.zeros((num_of_forecast,horizons))
all_mape_ts3=np.zeros((num_of_forecast,horizons))
all_msis_ts3=np.zeros((num_of_forecast,horizons))
for f in range(len(nums)):
    set_global_seed(seed_value_list[f])
    #set_global_seed(100)
    b=nums[f]
    e=b+train_len
    training_data=train_data.iloc[b:e,:]
    train_test_data=train_data.iloc[b:(e+horizons),:]
    filtered_data=pd.read_csv(filted_data_path+str(b+1)+'.csv').iloc[8:158,:]
    print('train_data_shape')
    print(training_data.shape)
    res_saving_path=saving_path+'section_'+str(b)+'/'
    # try:
        #model fitting
    lstm_model=train_network(training_data,filtered_data, num_layers, hidden_dim, iterations_trend, iterations_AR, m, order,lr,lr_trend, res_saving_path,threshould)
        #make predictions
    mse_list,mape_list,msis_list,point_forecast_array,final_lower_forecast_array,final_upper_forecast_array=forecast_based_on_pretrained_model(train_test_data,m,order,lstm_model,horizons,seasonality)
    all_mape_ts1[f,:]=mape_list[0]
    all_msis_ts1[f,:]=msis_list[0]
    all_mape_ts2[f,:]=mape_list[1]
    all_msis_ts2[f,:]=msis_list[1]
    all_mape_ts3[f,:]=mape_list[2]
    all_msis_ts3[f,:]=msis_list[2]
    #save forecasts
    pd.DataFrame(point_forecast_array).to_csv(res_saving_path+'point_forecasts.csv')
    pd.DataFrame(final_lower_forecast_array).to_csv(res_saving_path+'lower_forecasts.csv')
    pd.DataFrame(final_upper_forecast_array).to_csv(res_saving_path+'upper_forecasts.csv')

#calculate averaged accuracy
print('mape-ts1')
print(np.mean(all_mape_ts1,axis=0))
print('mape-ts2')
print(np.mean(all_mape_ts2,axis=0))
print('mape-ts3')
print(np.mean(all_mape_ts3,axis=0))

print('msis-ts1')
print(np.mean(all_msis_ts1,axis=0))
print('msis-ts2')
print(np.mean(all_msis_ts2,axis=0))
print('msis-ts3')
print(np.mean(all_msis_ts3,axis=0))





