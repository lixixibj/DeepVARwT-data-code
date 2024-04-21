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
from _model_fitting_for_real_data_no_filter import *
import re



# def get_t_function_values_(seq_len,horizon):
#     r"""
#         Get values of time functions.
#         Parameters
#         ----------
#         seq_len
#            description: the length of training data
#            type: int
#            shape: T

#         horizon
#            description: forecasting horizon
#            type: int
#            shape: h

#         Returns
#         -------
#         x
#            description: tensor of time function values
#            type: tensor
#            shape: (seq_len,batch,input_size)
#      """

#     x = np.zeros(shape=(seq_len+horizon,3))
#     t = (np.arange(seq_len+horizon) + 1) / seq_len
#     x[:,0] = t
#     x[:,1] = t * t
#     x[:,2] = t * t * t

#     x = torch.from_numpy(x.reshape((seq_len+horizon,1,3)))

#     return x




def get_t_function_values_(seq_len,num_of_t,horizon):
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

    x = np.zeros(shape=(seq_len+horizon,num_of_t))
    t = (np.arange(seq_len+horizon) + 1) / seq_len
    for i in range(num_of_t):
        x[:,i]=t**(i+1)

    x = torch.from_numpy(x.reshape((seq_len+horizon,1,num_of_t)))

    return x





def forecast_based_on_pretrained_model(train_test_data,order,pretrained_model_name,res_saving_path,horizon,seasonality,name_list):
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
    num_of_t,hidden_dim=get_num_of_t_and_h(pretrained_model_name)

    train_len = train_test_data.shape[0]-horizon
    m = data.shape[1]
    #x:shape(seq_len+horizon,batch,input_size)
    x=get_t_function_values_(train_len,num_of_t,horizon)
    #y: shape(T+h,m)
    y= torch.from_numpy(train_test_data.values)
    

    lstm_model = DeepVARwT(input_size=x.shape[2],
                          hidden_dim=hidden_dim,
                          num_layers=num_layers,
                          seqence_len=x.shape[1],
                           m=m,
                         order=order)
    lstm_model = lstm_model.float()

    lstm_model.load_state_dict(torch.load(res_saving_path+'pretrained_model/'+pretrained_model_name))
    train_test_len = x.shape[0]
    y_array = np.array(train_test_data)
    var_coeffs, residual_parameters, trend= lstm_model(x.float())

    #saving estimated trend
    # trend_list = []
    # for n in range(m):
    #     trend_flatten = torch.flatten(trend[:, :, n].t())
    #     trend_list.append(trend_flatten.tolist())
    # df_trend = pd.DataFrame(np.transpose(np.array(trend_list))) 
    # df_trend.to_csv(res_saving_path+'estimated_trend_using_mle.csv')  
    # #plot estimated trend
    # plot_estimated_trend(m,df_trend,train_test_data.iloc[0:train_len,:],res_saving_path+'trend/')
   

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
        print(mape_cal(acutal_observations[:,i], point_forecast_array[i,:]))
        ape[i,:]=mape_cal(acutal_observations[:,i], point_forecast_array[i,:])
        
        print(y_array[0:train_len,i])
        print(y_array[0:train_len,i].shape)
        print(acutal_observations[:,i])
        print(final_upper_forecast_array[i,:])
        print(final_lower_forecast_array[i,:])
        print('msis')
        print(msis_cal(y_array[0:train_len,i], acutal_observations[:,i],final_upper_forecast_array[i,:], final_lower_forecast_array[i,:], 0.05, seasonality, horizon))
        sis[i,:]=msis_cal(y_array[0:train_len,i], acutal_observations[:,i],final_upper_forecast_array[i,:], final_lower_forecast_array[i,:], 0.05, seasonality, horizon)


     #residual analysis
    trend_for_in_sample_data = torch.squeeze(trend[0:train_len, 0, :])
    de_trend_series=torch.from_numpy(train_test_data.iloc[0:train_len,:].values).float()-trend_for_in_sample_data
    pd.DataFrame(de_trend_series.detach().numpy()).to_csv(res_saving_path+'de_trend_series.csv')
    A_list = []
    for c in range(order):
        A_list.append(all_causal_coeffs[:, :, c])
    A_all = torch.cat(A_list, dim=1)
    #4.prepare lagged obs
    residual_m=torch.zeros([m, (train_len-order)])
    for t in range(order+1,train_len+1):
        print('t')
        print(t)
        p_lagged_data = []
        for i in range(order):
            obs=de_trend_series[(t - 2 - i), :].reshape(m, 1)
            p_lagged_data.append(obs)
        p_lagged_observations = torch.cat(p_lagged_data, dim=0)
        print('p_lagged_observations')
        print(p_lagged_observations.shape)
        #x_t: shape(m*1)
        residual=de_trend_series[(t-1), :].reshape(m, 1)-torch.mm(A_all,p_lagged_observations)
        print('residual-shape')
        print(residual.shape)
        residual_m[:,t-order-1]=residual[:,0]

    #begin to create acf_qq_plots file
    import os
    acf_qq_file_path = res_saving_path+'acf_qq/'
    folder = os.path.exists(acf_qq_file_path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(acf_qq_file_path)
    #name_list=['GDP gap','Inflation','Federal funds rate']   

    #begin to plot acf
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf
    import scipy.stats as stats
    import pylab
    for ts in range(m): 
        residual_ts = residual_m.detach().numpy()[ts,:]
        print('standard_residual')
        print(residual_ts.shape)
        print(residual_ts)
        #plot_acf(residual_ts, lags=50 , bartlett_confint=False)
        plot_acf(residual_ts, lags=30)
        plt.title(name_list[ts],fontsize=20,fontweight="bold")
        plt.xlabel("Lag",fontsize=15)
        plt.ylabel("Autocorrelation",fontsize=15)
        #plt.show()
        plt.xticks(np.arange(1, 30, 3))
        plt.savefig(acf_qq_file_path+str(ts)+'_auto_correlation_plots.png')
        plt.close()
#qq plot
        stats.probplot(residual_ts, dist="norm", plot=pylab)
        #pylab.show()
        plt.title(name_list[ts],fontsize=20,fontweight="bold")
        plt.xlabel("Theoretical quantiles",fontsize=15)
        plt.ylabel("Ordered values",fontsize=15)

        plt.savefig(acf_qq_file_path + str(ts) + '_qq_plots.png')
        plt.close()

    pd.DataFrame(residual_m.detach().numpy()).to_csv(acf_qq_file_path+'residual.csv')

    pytorch_total_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
    # print('total number of parameters')
    # print(pytorch_total_params)


    return ape,sis,point_forecast_array,final_lower_forecast_array,final_upper_forecast_array,pytorch_total_params









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



def get_num_of_t_and_h(input_string):
    # Extracting numbers using regular expressions
    t_match = re.search(r'_t_(\d+)', input_string)
    h_match = re.search(r'_h_(\d+)', input_string)
    # Check if matches are found and extract the numbers
    t_value = int(t_match.group(1)) if t_match else None
    h_value = int(h_match.group(1)) if h_match else None
    return t_value,h_value


from joblib import Parallel, delayed
import numpy as np
import pandas as pd

def process_section(f, train_len, horizon, num_of_t_list, num_of_h_list, num_layers, iter1, iter2, m, order, lr, lr_trend, threshould, data, saving_path, seasonality,name_list):
    b = f
    e = b + train_len
    training_data = data.iloc[b:e, :]
    train_test_data = data.iloc[b:(e + horizon), :]
    res_saving_path = saving_path + 'section_' + str(b) + '/'

    likelihood_list = []
    model_name_list = []

    for i, t_num in enumerate(num_of_t_list):
        for j, h_num in enumerate(num_of_h_list):
            set_global_seed(4000)
            likelihood, lstm_model, model_name = train_network(training_data, t_num, num_layers, h_num, iter1, iter2, m, order, lr, lr_trend, res_saving_path, threshould)
            likelihood_list.append(likelihood)
            model_name_list.append(model_name)

    min_index = likelihood_list.index(min(likelihood_list))
    optimal_model_name = model_name_list[min_index]

    ape, sis, forecast, fore_low, fore_upp,pytorch_total_params = forecast_based_on_pretrained_model(train_test_data, order, optimal_model_name, res_saving_path, horizon, seasonality,name_list)

    # save forecasts
    pd.DataFrame(forecast).to_csv(res_saving_path + 'point_forecasts.csv')
    pd.DataFrame(fore_low).to_csv(res_saving_path + 'lower_forecasts.csv')
    pd.DataFrame(fore_upp).to_csv(res_saving_path + 'upper_forecasts.csv')



    return ape, sis,pytorch_total_params



############begin to forecasting#################
import timeit
start = timeit.default_timer()
m=3
order=4
lr_trend=0.0005
lr=0.01
seasonality=4
name_of_dataset='./real-data/endog_data_m3_T193.csv'
#filted_data_path='./real-data/filtered-data/'
data=pd.read_csv(name_of_dataset).iloc[:,1:]
name_list=['GDP gap','Inflation','Federal funds rate']
print('data-shape')
print(data.shape)

len_of_data=data.shape[0]
hidden_dim=5
num_layers=1
iter1=2500
iter2=500
forecast_times=20
horizon=8
test_len=horizon+forecast_times-1
train_len=len_of_data-test_len
threshould=0.0000001


saving_path='./first-macro-data-python36-seed4000-iter1-'+str(iter1)+'-iter2-'+str(iter2)+'-h5-10-15-rep'+'/'



mape_all=np.zeros((forecast_times,m,horizon))
msis_all=np.zeros((forecast_times,m,horizon))


num_of_t_list=[2,3,4]
num_of_h_list=[5,10,15]

# Run the process in parallel
results = Parallel(n_jobs=-1)(delayed(process_section)(f, train_len, horizon, num_of_t_list, num_of_h_list, num_layers, iter1, iter2, m, order, lr, lr_trend, threshould, data, saving_path, seasonality,name_list) for f in range(forecast_times))

# Extract results
param_num=[]
for f, (ape, sis, num_of_params) in enumerate(results):
    mape_all[f, :, :] = ape
    msis_all[f, :, :] = sis
    param_num.append(num_of_params)

# Calculate averaged accuracy
print('MAPE')
mape=np.mean(mape_all, axis=0)
print(mape)
print('MAPE h1-4')
print(np.mean(mape[:,0:4], axis=1))
print('MAPE h1-8')
print(np.mean(mape, axis=1))

print('MSIS')
msis=np.mean(msis_all, axis=0)
print(msis)
print('MSIS h1-4')
print(np.mean(msis[:,0:4], axis=1))
print('msis h1-8')
print(np.mean(msis, axis=1))

print('num of parameters')
print(param_num)
print(min(param_num))
print(max(param_num))

stop = timeit.default_timer()


print('Time: ', stop - start) 



