1  #!/usr/bin/env python36
2  # -*- coding: utf-8 -*-


import numpy as np
import torch
import random
from custom_loss import *
from seed import *
#np.random.seed(0)
import random
from forecasting_accuracy import *
from lstm_network import *
from _model_fitting_for_real_data import *



def get_t_function_values_(seq_len,horizon):
    r"""
        Get values of time functions.
        Parameters
        ----------
        seq_len
           description: the length of training data
           type: int
           shape: T

        horizon
           description: forecasting horizon
           type: int
           shape: h

        Returns
        -------
        x
           description: tensor of time function values
           type: tensor
           shape: (seq_len,batch,input_size)
     """

    x = np.zeros(shape=(seq_len+horizon,3))
    t = (np.arange(seq_len+horizon) + 1) / seq_len
    x[:,0] = t
    x[:,1] = t * t
    x[:,2] = t * t * t

    x = torch.from_numpy(x.reshape((seq_len+horizon,1,3)))

    return x









def forecast_based_on_pretrained_model(train_test_data,order,pretrained_model,horizon,seasonality):
    r"""
        Network training.
        Parameters
        ----------
        train_test_data
           description: training and test data
           type: dataframe
           shape: (T+h,m)

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

    train_len = train_test_data.shape[0]-horizon
    m = data.shape[1]
    #x:shape(seq_len+horizon,batch,input_size)
    x=get_t_function_values_(train_len,horizon)
    #y: shape(T+h,m)
    y= torch.from_numpy(train_test_data.values)
    lstm_model = pretrained_model
    train_test_len = x.shape[0]
    y_array = np.array(train_test_data)
    var_coeffs, residual_parameters, trend= lstm_model(x.float())
    #begin to forecast
    # trend forecast shape(h,m)------>(m,h)
    trend_forecast=trend[train_len:,0,:].t()
    #last_p_trend shape(p,m)
    last_p_trend=trend[train_len-order:train_len,0,:]
    #lagged p observations
    p_lagged_value = []
    for i in range(order):
        obs = y_array[(train_len - 1 - i), :].reshape(m, 1)
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
    acutal_observations=y_array[train_len:,:]
    print('actual foreecat')
    print((acutal_observations))
    print('final forecast')
    print(point_forecast_array)
    ape=np.zeros((m,horizon))
    sis=np.zeros((m,horizon))


    for i in range(m):
        print('ts')
        print(i)
        print('actucal-forecast')
        print(acutal_observations[:, i])
        print('mape')
        #print(mape_cal(acutal_observations[:,i], point_forecast_array[i,:]))
        ape[i,:]=mape_cal(acutal_observations[:,i], point_forecast_array[i,:])
        print('msis')
        print(y_array[0:train_len,i])
        print(y_array[0:train_len,i].shape)
        print(acutal_observations[:,i])
        print(final_upper_forecast_array[i,:])
        print(final_lower_forecast_array[i,:])
        print(msis_cal(y_array[0:train_len,i], acutal_observations[:,i],final_upper_forecast_array[i,:], final_lower_forecast_array[i,:], 0.05, seasonality, horizon))
        sis[i,:]=msis_cal(y_array[0:train_len,i], acutal_observations[:,i],final_upper_forecast_array[i,:], final_lower_forecast_array[i,:], 0.05, seasonality, horizon)


    return ape,sis,point_forecast_array,final_lower_forecast_array,final_upper_forecast_array









def cal_var_cov_of_prediction_error(A,residual_parameters,horizon,order,m):
    r"""
        Calculate variance-covariance matrix of prediction error of different horizons
        Parameters
        ----------
        A
           description: coefficient matrix of VAR(1)
           type: tensor
           shape: (m*p,m*p)

      test1  residual_parameters
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
order=4
lr_trend=0.0005
lr=0.01
seasonality=4
name_of_dataset='./real-data/endog_data_m3_T193.csv'
filted_data_path='./real-data/filtered-data/'
data=pd.read_csv(name_of_dataset).iloc[:,1:]
print('data-shape')
print(data.shape)

len_of_data=data.shape[0]
hidden_dim=20
num_layers=1
iter1=4000
iter2=8000
forecast_times=20
horizon=8
test_len=horizon+forecast_times-1
train_len=len_of_data-test_len
threshould=0.000005

saving_path='./real-data-forecasting-res-test3/'

seed_value_list=[2100,0,4000,4000,4000,
                 4000,4000,4000,4000,4000,
                 4000,4000,4000,4000,4000,
                 4000,500,4000,4000,4000]

mape_all=np.zeros((forecast_times,m,horizon))
msis_all=np.zeros((forecast_times,m,horizon))

for f in range(forecast_times):
    set_global_seed(seed_value_list[f])
    #set_global_seed(100)
    b=f
    e=b+train_len
    training_data=data.iloc[b:e,:]
    train_test_data=data.iloc[b:(e+horizon),:]
    filtered_data=pd.read_csv(filted_data_path+str(b+1)+'.csv').iloc[8:158,:]
    print('train_data_shape')
    print(training_data.shape)
    res_saving_path=saving_path+'section_'+str(b)+'/'
    # try:
        #model fitting
    lstm_model=train_network(training_data,filtered_data, num_layers, hidden_dim, iter1, iter2, m, order,lr,lr_trend, res_saving_path,threshould)
        #make predictions
    ape,sis,forecast,fore_low,fore_upp=forecast_based_on_pretrained_model(train_test_data,order,lstm_model,horizon,seasonality)
    mape_all[f,:,:]=ape
    msis_all[f,:,:]=sis
    #save forecasts
    pd.DataFrame(forecast).to_csv(res_saving_path+'point_forecasts.csv')
    pd.DataFrame(fore_low).to_csv(res_saving_path+'lower_forecasts.csv')
    pd.DataFrame(fore_upp).to_csv(res_saving_path+'upper_forecasts.csv')

#calculate averaged accuracy
print('MAPE')
print(np.mean(mape_all,axis=0))
print('MSIS')
print(np.mean(msis_all,axis=0))

