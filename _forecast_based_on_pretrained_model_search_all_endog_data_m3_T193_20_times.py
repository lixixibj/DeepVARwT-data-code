1  #!/usr/bin/env python35
2  # -*- coding: utf-8 -*-
3  # @File  : data_processing.py
4  # @Author: Xixi Li
5  # @Date  : 2019-12-02
6  # @Desc  :


from custom_loss import *
import numpy as np
import pandas as pd
np.random.seed(0)

from time_feature import time_feature
import numpy as np
import torch
import torch.utils.data as Data
import random
from custom_loss import *
from min_max_transformation import *
import os
from forecasting_accuracy import *






#add plolynomial
#ts_len: training+test
def get_time_function(ts_len,horizon):
    #init np for saving time feature n*m
    #len_of_ts = len(train_entry['target'])
    len_of_ts=ts_len-horizon
    # print(ts_entry)
    # print('len_of_ts')
    # print(len_of_ts)
    #time_feature_array=np.zeros(shape=(1,len_of_ts))
    time_feature_array = np.zeros(shape=(3, ts_len))
    for i in range(ts_len):
        t=(i+1)/len_of_ts
        time_feature_array[0,i]=t
        time_feature_array[1, i] =t*t
        time_feature_array[2, i] = t * t*t
        # inverse_t=1/(i+1)
        # time_feature_array[3,i]=inverse_t
        # time_feature_array[4, i] =inverse_t*inverse_t
        # time_feature_array[5, i] = inverse_t * inverse_t*inverse_t

        #we can add some polynomial


    #ts_entry['time_feature']=time_feature_array
    # print('time_feature')
    # print(time_feature_array)

    return time_feature_array




#constract feature for multivaraite data eg:begin_position=10,10+window_size,
#all_train_data: initial train data from gluon-ts
#features: shape: seq_t*number_of_features
#return: final feature of one multivarite ts: shape: seq_t*num.of.features
#begin_time:"10-05-1991",
#Freq=D
def get_time_function_for_input(train_test_data,horizon):
    #get data
    train_data = {}
    #data_name = name_of_dataset + '_train.csv'
    #data = pd.read_csv(data_name)
    data=train_test_data
    seq_len=data.shape[0]
    #time_feature_array: shape(covariates*seq_len)
    time_feature_array=get_time_function(seq_len,horizon)
    observations=data.iloc[:,1:]
    #observations: shape(m*seq_len)
    observations=np.array(observations).T
    print('observations')
    print(observations.shape)
    time_feature_temp=time_feature_array
    time_feature_array1=time_feature_temp.transpose().tolist()
    time_features=[]
    time_features.append(time_feature_array1)
    covairates_array= np.array(time_features)
    # original_shape: num.of.train.data*seq_t*num.of.features (batch,seq,input)
    # here we need to change the shape of the all_train_data_feature to(sep,batch,input)
    train_data['t_functions'] = torch.from_numpy(covairates_array)
    # shape: num.of.train.data*multivarite*seq_t
    # train_data['multi_target']=nd.array(np.array(all_train_data_target_list), dtype=nd.float64)
    #observations shape(seq_len*m)
    observations=data.iloc[:,1:]
    observations_array=np.array(observations)
    #observations_array: shape(1*seq_len*m)
    train_data['multi_target'] = torch.from_numpy(observations_array)

    return train_data





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







from var_ssm import VARSSM
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas
#training nework


from lstm_network import *
def forecast_based_on_pretrained_model(train_test_data,num_layers,hidden_dim,m,order,
                                       path_of_pretrained_model,horizon,seasonality):
    # process with real data 20200204
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(0)
    #here we need to remove mean of the original of ts(training part not contains test
    train_data = get_time_function_for_input(train_test_data,horizon)

    #train_data = get_train_data_parellel(all_train_data, sequence_length, feature_list, time_feature_list,overlap,num_of_train_data)
    x = train_data['t_functions']
    print('x.shape')
    print(x.shape)
    y = train_data['multi_target']
    sequence_length = x.shape[1]
    lstm_model = DeepVARwT(input_size=x.shape[2],
                          hidden_dim=hidden_dim,
                          num_layers=num_layers,
                          seqence_len=sequence_length,
                           m=m,
                           order=order)
    lstm_model = lstm_model.float().to(device)
    PATH = path_of_pretrained_model
    #checkpoint = torch.load(PATH)
    lstm_model.load_state_dict(torch.load(PATH))
    count_temp=1
    seqence_len_all = x.shape[1]
    seqence_len_train=seqence_len_all-horizon
    print('seqence_len_train')
    print(seqence_len_train)

    #data_name = name_of_dataset + '_train.csv'
    #data = pd.read_csv(data_name)
    data=train_test_data
    #observations: shape(T*m)
    observations = np.array(data.iloc[:, 1:])
    x_input = change_data_shape(x)
    var_coeffs, redusial_parameters, trend_t= lstm_model(x_input.float().to(device))
        #begin to forecast
        # trend forecast shape(horizon*m)------(m*horizon)
    trend_forecast=trend_t[seqence_len_train:,0,:].t()
    print('trend forecast')
    print(trend_t.shape)
        #last_p_trend shape(p*m)
    last_p_trend=trend_t[seqence_len_train-order:seqence_len_train,0,:]
    #last p trend
    p_lagged_value = []
    for i in range(order):
        obs = observations[(seqence_len_train - 1 - i), :].reshape(m, 1)
        detrend_obs=torch.from_numpy(obs).float()-last_p_trend[(order-1-i),:].reshape(m,1)
        p_lagged_value.append(detrend_obs)
    p_lagged_observations = torch.cat(p_lagged_value, dim=0)
    mp = m * order

        #here A is fixed
    var_cov_innovations_varp = make_var_cov_matrix_for_innovation_of_varp(redusial_parameters, m, order)
    all_stationary_coeffs = A_coeffs_for_causal_VAR(var_coeffs, order, m, var_cov_innovations_varp)
    #FF = transform_data_size_with_tensor_latest(all_stationary_A, m, lag_order)
    A_coeffs_var1 = get_A_coeff_m_for_VAR_1(all_stationary_coeffs, m, order)


    A_list = []
    for c in range(order):
        A_list.append(all_stationary_coeffs[:, :, c])
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
        print('forecasts_part_shape')
        print(forecasts_part.shape)
        #forecasts=torch.mm(A_all, p_lagged_observations)

        #updated_lagged_value=[forecasts,p_lagged_observations]
        #p_lagged_observations=torch.cat(updated_lagged_value,dim=0)[0:mp,0].reshape(mp,1)
        forecasts_list.append(forecasts_part)
            #cal interval prediction
        square_root_list=cal_var_cov_of_prediction_error(A_coeffs_var1,redusial_parameters,(t+1),order,m)
        print('sqare-root')
        print(square_root_list)
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
    final_forecast_array=final_forecast.detach().numpy()
    final_upper_forecast_array = final_upper_forecast.detach().numpy()
    final_lower_forecast_array = final_lower_forecast.detach().numpy()
        #acutal_observations:shape(horizon,m)
    acutal_observations=observations[seqence_len_train:,:]
    print('actual foreecat')
    print((acutal_observations))
    print('final forecast')
    print(final_forecast_array)
    mse_list = []
    mape_list = []
    msis_list = []


    for i in range(m):
        print('ts')
        print(i)
        print('actucal-forecast')
        print(acutal_observations[:, i])
        print('mse')
        print(mse_cal(acutal_observations[:, i], final_forecast_array[i, :]))
        mse_list.append(list(mse_cal(acutal_observations[:, i], final_forecast_array[i, :])))
        print('mape')
        print(mape_cal(acutal_observations[:,i], final_forecast_array[i,:]))
        mape_list.append(mape_cal(acutal_observations[:,i], final_forecast_array[i,:]))
        print('msis')
        print(msis(observations[0:seqence_len_train,i], acutal_observations[:,i],
                       final_upper_forecast_array[i,:], final_lower_forecast_array[i,:], 0.05, seasonality, horizon))
        msis_list.append(msis(observations[0:seqence_len_train,i], acutal_observations[:,i],
                       final_upper_forecast_array[i,:], final_lower_forecast_array[i,:], 0.05, seasonality, horizon))

    return mse_list,mape_list,msis_list,final_forecast_array,final_lower_forecast_array,final_upper_forecast_array





        ########interval forecasting
















#inteval prediction for each
#here Q_coeffs, shape:(seq,batch,m*(m+1)/2), we need mp*mp matrix
#horizon: (1----h)
def cal_var_cov_of_prediction_error(A,residual_parameters,horizon,order,m):

    mp=m*order
    identy_m=torch.eye(m)
    zeros_cols = torch.zeros([m, (mp - m)])
    #J: shape(k,kp)
    J= torch.cat((identy_m, zeros_cols), 1)

    # U_covariance:shape(mp*mp)
    U_covariance = make_covariance_matrix_Q_t(residual_parameters, m, order)

    var_cov_temp=torch.zeros(mp,mp)
    print('A-shape')
    print(A.shape)
    for i in range(horizon):
        A_multip=multipy_A_matrix(A,i,mp)
        var_cov_temp=var_cov_temp+torch.mm(torch.mm(A_multip,U_covariance),A_multip.t())

        #J_multiply_FF=torch.mm(J,multipy_A_matrix(A,i,mp))
        #var_cov=var_cov+torch.mm(torch.mm(J_multiply_FF,U_covariance),J_multiply_FF.t())
    var_cov=torch.mm(torch.mm(J,var_cov_temp),J.t())
    #cal h_step forecast variance of each time series
    var_list=[]
    import math
    for n in range(m):
        var_list.append(math.sqrt(var_cov[n,n]))
    return var_list


#calculate FF^{i} power law of matrix
def multipy_A_matrix(FF,i,mp):
    A_multiply_tmp = torch.eye(mp, mp)
    if i==0:
        return torch.eye(mp, mp)
    else:
        for num in range(i):
            A_multiply_tmp=torch.mm(A_multiply_tmp,FF)
        return A_multiply_tmp



def get_number_of_forecast(file_name):
    str_list=file_name.split('_')
    # print(str_list)
    # print(str_list[7][0:2])
    str_num=str_list[7][:-5]
    return int(str_num)

def get_stop_num(threshould,likelihood_list):
    trigger=0
    for num in range(len(likelihood_list)-2):
        current_likelihood=-likelihood_list[num]
        furture_likelihood=-likelihood_list[num+1]
        abs_relative_error1=abs((furture_likelihood-current_likelihood)/current_likelihood)
        print('abs_relative_error1')
        print(abs_relative_error1)
        if abs_relative_error1<threshould:
            trigger=trigger+1
            current_likelihood=-likelihood_list[num+1]
            furture_likelihood=-likelihood_list[num+2]
            abs_relative_error2=abs((furture_likelihood-current_likelihood)/current_likelihood)
            print('abs_relative_error2')
            print(abs_relative_error2)
            if abs_relative_error2<threshould:
                print('trigger')
                trigger=trigger+1
                print('num-index')
                print(num)
                break
    return (num+2)



############begin to forecasting#################
m=3
order=2
lr_trend=0.0001
lr_trend=0.001
lr_ar=0.01
seasonality=4
name_of_dataset='/Users/xixili/Dropbox/deep-var-tend/deep-factors/data/endog_data_m3_T193.csv'
train_data=pd.read_csv(name_of_dataset)

len_of_data=train_data.shape[0]
hidden_dim=20
num_layers=1
iterations_trend=4000
iterations_AR=10000
print('hidden-dim')
print(hidden_dim)
num_of_forecast=20
horizons=8
test_len=horizons+num_of_forecast-1

train_len=len_of_data-test_len

ranges_list=[9950]
ranges_list=range(100,iterations_AR,50)
len_r=len(ranges_list)


thresould_list=[0.00001,0.00002]






thresould_list=[0.000002,0.05,0.005,0.0005,0.00005,0.000005]
thresould_list=[0.000002]

name_of_train_dataset = str.split(str.split(name_of_dataset, '/')[-1],'_')[0]
path =name_of_train_dataset+ '_h' + str(hidden_dim) + '_trend_iters_' + str(iterations_trend) + '_AR_iters_' + str(iterations_AR)+'_lr_t'+str(lr_trend)+'_lr_ar'+str(lr_ar)+'/'
res_path='/Users/xixili/Dropbox/deep-var-tend/deep-factors/real-data/endog-forecast-many-times-p2-lrtrend001-same-seed-400/3-t-h20/'
accuracy_saving_path='/Users/xixili/Dropbox/deep-var-tend/deep-factors/real-data/endog-forecast-many-times-p2-lrtrend001-same-seed-400/accuracy-test-new/'



for thred in thresould_list:
        #save
    accuracy_path=accuracy_saving_path+'3-t-h20/'+str(thred)+'two_thre'+'/'
    folder = os.path.exists(accuracy_path)
    if not folder:
        os.makedirs(accuracy_path)
    #arrray
    mape_accuracy_one_thred=[np.zeros((len_r,horizons+2)),np.zeros((len_r,horizons+2)),np.zeros((len_r,horizons+2))]
    msis_accuracy_one_thred=[np.zeros((len_r,horizons+2)),np.zeros((len_r,horizons+2)),np.zeros((len_r,horizons+2))]
    for r in range(len_r):
        ranges=ranges_list[r]
        #best_fitting_model_index=[]
        #20 times
        #get all 20 times of forecast
        import os
        all_file_forecast=os.listdir(res_path)
        #get number of forecast
        all_mape_ts=[np.zeros((num_of_forecast,horizons)),np.zeros((num_of_forecast,horizons)),np.zeros((num_of_forecast,horizons))]
        all_msis_ts=[np.zeros((num_of_forecast,horizons)),np.zeros((num_of_forecast,horizons)),np.zeros((num_of_forecast,horizons))]

        for file_name in all_file_forecast:
            num=get_number_of_forecast(file_name)
            #get likelihood
            likelihood_path=res_path+file_name+'/likelihood_loss.csv'
            likelihood=pd.read_csv(likelihood_path).iloc[:,1].tolist()[0:ranges]
            #get index accoding to threshould and maximum training times
            best_fitting_model_index=get_stop_num(thred,likelihood)+iterations_trend
            print('ts-section')
            print(num)
            print('best_fitting_model_index')
            print(best_fitting_model_index)
            model_path = res_path+file_name + '/pretrained_model/'
            model_saving_path = model_path + str(best_fitting_model_index) + '_net_params.pkl'
            #get train and test data
            b=num
            e=b+train_len+horizons
            section_data=train_data.iloc[b:e,:]
            mse_list, mape_list, msis_list, point_forecats, lower_forecasts, upper_forecasts = forecast_based_on_pretrained_model(section_data, num_layers, hidden_dim, m, order,model_saving_path, horizons, seasonality)
            for ts in range(m):
                all_mape_ts[ts][num,:]=mape_list[ts]
                all_msis_ts[ts][num,:]=msis_list[ts]
        #calculate averging 20 times
        for ts in range(m):
            all_mape=all_mape_ts[ts]
            mean_h1_4_mape=all_mape[:,0:4].mean(axis=1)
            mean_h1_8_mape=all_mape.mean(axis=1)
            mape_array=np.c_[all_mape, mean_h1_4_mape,mean_h1_8_mape]  
            #averaging 20 times
            mean_mape= mape_array.mean(axis=0)
            mape_accuracy_one_thred[ts][r,:]=mean_mape

            all_msis=all_msis_ts[ts]
            mean_h1_4_msis=all_msis[:,0:4].mean(axis=1)
            mean_h1_8_msis=all_msis.mean(axis=1)
            msis_array=np.c_[all_msis, mean_h1_4_msis,mean_h1_8_msis]  
                        #averaging 20 times
            mean_msis= msis_array.mean(axis=0)
            msis_accuracy_one_thred[ts][r,:]=mean_msis
    if r==(len_r-1):
        for ts in range(m):
            pd.DataFrame(all_mape_ts[ts]).to_csv(accuracy_path+str(ts)+'_mape_20_times.csv')
            pd.DataFrame(all_msis_ts[ts]).to_csv(accuracy_path+str(ts)+'_msis_20_times.csv')
        







    for ts in range(m):
        pd.DataFrame(mape_accuracy_one_thred[ts]).to_csv(accuracy_path+str(ts)+'_mape.csv')
        pd.DataFrame(msis_accuracy_one_thred[ts]).to_csv(accuracy_path+str(ts)+'_msis.csv')







