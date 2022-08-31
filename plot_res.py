1  # !/usr/bin/env python35
2  # -*- coding: utf-8 -*-
3  # @File  : data_processing.py
4  # @Author: Xixi Li
5  # @Date  : 2019-12-02
6  # @Desc  :

# -----------------main code for simulation study------------------------


import numpy as np
import pandas as pd

np.random.seed(0)
import torch.utils.data as Data
import random
from custom_loss import *
from var_ssm import VARSSM
from seed import *
import random
seed_value=random.randint(2, 1000)
from lstm_network import*
seed_value=54
seed_value=81
set_global_seed(seed_value)
import torch
from scipy.spatial import distance



# here we just use time functions as our inuts
# add plolynomial
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

    time_functions_array = np.zeros(shape=(6, series_len))
    t = (np.arange(series_len) + 1) / series_len
    time_functions_array[0, :] = t
    time_functions_array[1, :] = t * t
    time_functions_array[2, :] = t * t * t
    inverse_t = 1 / (np.arange(series_len) + 1)
    time_functions_array[3, :] = inverse_t
    time_functions_array[4, :] = inverse_t * inverse_t
    time_functions_array[5, :] = inverse_t * inverse_t * inverse_t

    return time_functions_array


def get_data_and_time_function_values(file_of_dataset):
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
    data_name = file_of_dataset + '_train.csv'
    data = pd.read_csv(data_name)
    seq_len = data.shape[0]
    train_data = {}
    multivarite_ts = {}
    # time_feature_array: shape(6*seq_len)
    time_functions_array = get_t_function_values_(seq_len)
    # here add observations
    # observations shape(seq_len*m)
    observations = data.iloc[:, 1:]
    # observations: shape(m*seq_len)
    observations = np.array(observations).T
    # print('observations')
    # print(observations.shape)
    time_functions_temp = time_functions_array
    time_functions_array1 = time_functions_temp.transpose().tolist()
    time_functions = []
    time_functions.append(time_functions_array1)
    t_func_array = np.array(time_functions)
    # original_shape: num.of.train.data*seq_t*num.of.features (batch,seq,input)
    # here we need to change the shape of the all_train_data_feature to(sep,batch,input)
    train_data['t_functions'] = torch.from_numpy(t_func_array)
    # shape: num.of.train.data*multivarite*seq_t
    # train_data['multi_target']=nd.array(np.array(all_train_data_target_list), dtype=nd.float64)
    # observations shape(seq_len*m)
    observations = data.iloc[:, 1:]
    observations_array = np.array(observations)
    # observations_array: shape(seq_len*m)
    train_data['multi_target'] = torch.from_numpy(observations_array)

    return train_data


# original data: shape(batch,seq,input)
# changed data: shape(seq,batch,input)
def change_data_shape(original_data):
    # change to numpy from tensor
    original_data = original_data.numpy()
    new_data = []
    for seq_temp in range(original_data.shape[1]):
        new_data.append(original_data[:, seq_temp, :].tolist())
    # change to tensor
    new_data = torch.from_numpy(np.array(new_data))
    return new_data


import ray
import time


#@ray.remote
def plot_estimated_res(file_of_dataset, num_layers, hidden_dim, m, order,model_path):
    r"""
        Train neural network
        Parameters
        ----------
        file_of_dataset
           description: the path of data file
           type: str
        num_layers
           description: the number of layer of lstm
           type: int
        epoches
           description: the number of iterations
           type: int
        hidden_dim
           description: the number of dimenison of hidden state in lstm
           type: int
        m
           description: the dimension of multivariate ts
           type: int
        order
           description: the order of VAR
           type: int
        simulated_A_path
           description: the path of simulated A coeffs
           type: str
        simulated_lower_tri_path
           description: the path of simulated lower tri path
           type: str
        Returns
        -------
    """

    import torch
    # here we need to remove mean of the original of ts(training part not contains test
    train_data = get_data_and_time_function_values(file_of_dataset)

    # train_data = get_train_data_parellel(all_train_data, sequence_length, feature_list, time_feature_list,overlap,num_of_train_data)
    x = train_data['t_functions']
    # print('t_functions')
    # print(x)
    sequence_length = x.shape[1]
    y = train_data['multi_target']
    # print('target')
    # print(y)
    from lstm_network import DeepVARwT
    lstm_model = DeepVARwT(input_size=x.shape[2],
                          hidden_dim=hidden_dim,
                          num_layers=num_layers,
                          seqence_len=sequence_length,
                          m=m,
                          order=order
                          )
    lstm_model = lstm_model.float()
    #checkpoint = torch.load(PATH)
    lstm_model.load_state_dict(torch.load(model_path))


    #optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)

    count_temp = 1
    x_input = change_data_shape(x)
    print('x-input')
    print(x_input.shape)

    var_coeffs, redusial_parameters,trend_t = lstm_model(x_input.float())
    #get A
      # all_stationary_coeffs = staionary_coefs(initial_var_coeffs=var_coeffs, p=lag_order, d=m)
    var_cov_innovations_varp = make_var_cov_matrix_for_innovation_of_varp(redusial_parameters, m, order)
    all_stationary_coeffs = A_coeffs_for_causal_VAR(var_coeffs, order, m, var_cov_innovations_varp)
    coeffs_list=[]
    for i in range(order):
        A=all_stationary_coeffs[:, :, i].reshape((1,m*m)).tolist()[0]
        # print('A')
        # print(all_stationary_coeffs[:, :, i].reshape((1,m*m))[0])
        # print(A)
        coeffs_list.extend(A)
    var_cov_list=var_cov_innovations_varp.reshape((1,m*m)).tolist()[0]
    #trend
    trend_list = []
    for n in range(y.shape[1]):
            # one sub-seqence
        trend_flatten = torch.flatten(trend_t[:, :, n].t())
        trend_list.append(trend_flatten.tolist())
    trend = np.transpose(np.array(trend_list))
    # print('coeffs_list')
    # print(coeffs_list)
    # print(var_cov_list)
    print('trend-shape')
    print(trend.shape)



    return coeffs_list, var_cov_list, trend





def get_number_of_forecast(file_name):
    str_list=file_name.split('_')
    # print(str_list)
    # print(str_list[7][0:2])
    str_num=str_list[2][3:]
    return int(str_num)

import os


name_of_dataset='/Users/xixili/Dropbox/DeepTVAR-code/simulation-T200/R-parameters/tvvar_p2_m2_T200_causality'

m = 3
order = 2
hidden_dim=20

num_layers = 1
epoches =1000



res_path='/Users/xixili/Dropbox/deep-var-tend/deep-factors/sim_exp100/res-len800-var2-m3/'

threshould=1e-5
plot_res_path='/Users/xixili/Dropbox/deep-var-tend/deep-factors/sim_exp100/plots/T800-exp100-h20-lstm-final/'+str(threshould)+'loss_iters6000_100_times'+'/'



num_layers=1
iterations_trend=15000
iterations_AR=6000

data_saving_path='/Users/xixili/Dropbox/deep-var-tend/deep-factors/sim_exp100/simulated_data_len800_VAR2_m3/'
#saving_path='./image/simulation-estimation-res-two-steps-diff-lr/p2_k3'
res_saving_path='/Users/xixili/Dropbox/deep-var-tend/deep-factors/sim_exp100/res-len800-var2-m3/'
simulated_trend_path='/Users/xixili/Dropbox/deep-var-tend/deep-factors/trend_from_real_data/trend-len800-kernal-for-varwT.csv'

len_of_seq=800
all_A_coffs=np.zeros((100, (1+m * m * order)))
all_var_cov_elemnts = np.zeros((100, (1+m*m)))
trend_all= np.zeros((len_of_seq, m))
h=((m+1)*m)/2
all_lower_elemnts = np.zeros((int(h), len_of_seq))
trend_list=[]


final_loss_value_list=[]
remove_list=[]
import os
all_file=os.listdir(res_path)
#name_of_dataset=data_saving_path+str(1)+'_TVAR_m2_p2_T2'
name_of_dataset=data_saving_path+'exp'+str(1)+'_len800_VAR2_m3'

num_of_coeff_parameters=m*m*order
num_of_var_cov_parameters=m*m
count_num=0
exp_num=[]
seed_number=[]
iterations=[]
valid_count=0
remove_index=[1,6,9,11,13,15,18,19,20,21,
              26,29,31,33,36,39,41,43,45,48,
              49,50,51,56,59,60,61,65,66,69,
              71,73,75,77,78,79,81,82,86,89,
              91,95,96,99,100,101,102,103,105,108,
              109,110,111,116,119,121,123,126,129,131,
              133,135,138,139,140,141,142,147,149,151,
              156,160,161,166,171]

#get num 
for i in range(len(all_file)): 
    #get
    exp_index=get_number_of_forecast(all_file[i])

    if exp_index in remove_index:
        print('exp_index')
        print(exp_index)
        continue
    path=res_path+all_file[i]+'/'
    loss_df=pd.read_csv(path+'likelihood_loss.csv')
    abs_relative_error1=1
    abs_relative_error2=1
    abs_relative_error3=1
    trigger=0
    nums=loss_df.shape[0]
    #for num in range(nums-3):
    for num in range(2,6000):
        current_loss=loss_df.iloc[num,1]
        past1_loss=loss_df.iloc[num-1,1]
        abs_relative_error1=abs((current_loss-past1_loss)/past1_loss)
        # print('abs_relative_error1')
        # print(abs_relative_error1)
        if abs_relative_error1<threshould:
            trigger=trigger+1
            past1_loss=loss_df.iloc[num-1,1]
            past2_loss=loss_df.iloc[num-2,1]
            abs_relative_error2=abs((past1_loss-past2_loss)/past2_loss)
            # print('abs_relative_error2')
            # print(abs_relative_error2)
            
            if abs_relative_error2<threshould:
                #print('trigger')
                trigger=trigger+1
                # print('num-index')
                # print(num)
                break
                # current_loss=lines_array[num+2]
                # furture_loss=lines_array[num+3]
                # abs_relative_error3=abs((furture_loss-current_loss)/current_loss)
                # if abs_relative_error3>threshould:
                #     trigger=trigger+1
                #     break

    count_temp=num
    # print('num')
    # print(num)
    # print('len')
    final_loss=loss_df.iloc[num,1]
    #if final_loss<-240 or  or math.isnan(final_loss):
    # if math.isnan(final_loss) or final_loss<-150 :
    #     continue
    exp_num.append(all_file[i].split('_')[2])
    seed_number.append(all_file[i].split('_')[0])
    iterations.append(count_temp)
    final_loss_value_list.append(final_loss)
    model_path=path+'pretrained_model/'+str(count_temp+iterations_trend)+'_net_params.pkl'
    #plot_estimated_res(file_of_dataset, num_layers, epoches, hidden_dim, m, order,model_path)
    coeffs, var_cov, trend= plot_estimated_res(name_of_dataset, num_layers, hidden_dim, m, order,model_path)
    all_A_coffs[valid_count,0]=exp_index
    all_A_coffs[valid_count,1:]=coeffs
    all_var_cov_elemnts[valid_count,0]=exp_index
    all_var_cov_elemnts[valid_count,1:]=var_cov
    trend_all=trend_all+trend
    trend_list.append(trend)

    valid_count=valid_count+1


# all_A_coffs_valid=all_A_coffs[0:valid_count,]
# all_var_cov_elemnts_valid=all_var_cov_elemnts[0:valid_count,]

all_A_coffs_valid=all_A_coffs
all_var_cov_elemnts_valid=all_var_cov_elemnts

print('valid-times')
print(valid_count)
print('estimated-A-mean')
print(np.mean(all_A_coffs_valid, axis=0))

print('estimated-A-var')
print(np.var(all_A_coffs_valid, axis=0))
print('estimated-var-mean')
print(np.mean(all_var_cov_elemnts_valid, axis=0))

print('estimated-var-var')
print(np.var(all_var_cov_elemnts_valid, axis=0))
mean_trend=trend_all/valid_count

squred_error_trend_sum=np.zeros((len_of_seq, m))

for i in range(valid_count):
    diff=trend_list[i]-mean_trend
    squred_error=np.multiply(diff,diff)
    squred_error_trend_sum=squred_error_trend_sum+squred_error

print('np.sqrt(n)')
print(np.sqrt(valid_count))

sd_trend=1.96*(np.sqrt(squred_error_trend_sum/(valid_count-1))/np.sqrt(valid_count))
sd_trend_df=pd.DataFrame(sd_trend)


import pandas as pd
simulated_trend_df = pd.read_csv(simulated_trend_path)
estimated_trend_df = pd.DataFrame(mean_trend)
trend_path = plot_res_path+'/trend/'
trend_folder = os.path.exists(trend_path)
if not trend_folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(trend_path)
estimated_trend_df.to_csv(trend_path+ 'trend_estimated_mean' + '.csv')

from matplotlib.pylab import plt
plt.style.use('classic')
            # %matplotlib inline
k = simulated_trend_df.shape[1]-1
x_value = np.array(range(simulated_trend_df.shape[0]))
# print('x-value')
# print(x_value.shape)
lower_trend= np.zeros((len_of_seq, m))
upper_trend= np.zeros((len_of_seq, m))

for i in range(k):
                # plt.figure(figsize=(20, 10))
    s = np.array(simulated_trend_df.iloc[:,(i+1)])
                # plt.plot(s)
    e = np.array(estimated_trend_df.iloc[:, i])
    #e_lower=
    sd=np.array(sd_trend_df.iloc[:,i])
    l=e-sd
    u=e+sd
    lower_trend[:,i]=l
    upper_trend[:,i]=u



    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(x_value, list(s), '-k', label='Simulation')
    ax.plot(x_value, list(e), '-r', label='Estimation')
    ax.plot(x_value, list(l), '-g', label='Lower')
    ax.plot(x_value, list(u), '-m', label='Upper')

    # ax.plot(x_value, list(e-np.array(sd_A.iloc[i,order:])), '-g', label='Sample lower sd')
    # ax.plot(x_value, list(e+np.array(sd_A.iloc[i,order:])), '-g', label='Sample uppder sd')
    #ax.plot(x_value, list(u), '-m', label='Sample upper')
    # ax.plot(x_value, list(asyp_l), '-y', label='Asymptotic lower')
    # ax.plot(x_value, list(asyp_u), '-b', label='Asymptotic  upper')
                # ax.axis('equal')
                # leg = ax.legend();
    ax.legend(loc='upper left', frameon=False)

    name = str(i + 1) + '_p.png'
                # plt.savefig('./image/poly_trend_sine_s50/A/' + name)
    plt.savefig(trend_path + '/' + name)
    plt.close()

pd.DataFrame(lower_trend).to_csv(trend_path+ 'trend_estimated_lower' + '.csv')
pd.DataFrame(upper_trend).to_csv(trend_path+ 'trend_estimated_upper' + '.csv')

print('final_loss_value_list')
print(final_loss_value_list)
print('iterations')
print(iterations)



pd.DataFrame(all_A_coffs_valid).to_csv(plot_res_path+'all-A.csv')
pd.DataFrame(all_var_cov_elemnts_valid).to_csv(plot_res_path+'all-var-cov.csv')
