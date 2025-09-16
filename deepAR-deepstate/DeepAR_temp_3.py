1  #!/usr/bin/env python36
2  # -*- coding: utf-8 -*-



from gluonts.dataset import common
from gluonts.model import deepstate
from forecasting_accuracy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import ray
from tqdm.autonotebook import tqdm
from itertools import islice
import matplotlib as mpl
from seed import*
set_global_seed(2)

from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer

def plot_forecasts(tss, forecasts, past_length, num_plots,saving_path):
    i=0
    for target, forecast in islice(zip(tss, forecasts), num_plots):
        mpl.use('tkagg')
        prediction_intervals = [95]
        ax = target[-past_length:].plot(figsize=(12, 5), linewidth=2)
        forecast.plot(prediction_intervals=prediction_intervals,color='g')
        plt.grid(which='both')
        plt.legend(["observations", "median prediction", "95% confidence interval"])
        #plt.show()
        path=saving_path+'ts'+str(i)+'.png'
        plt.savefig(path)
        plt.close()
        i=i+1

def deepAR(data,frequency,seasonality,h,starting_time,figure_saving_path):
    #data: shape(time*(m+1))
    len_all = data.shape[0]
    train_data = data.iloc[0:len_all - h, 1:]
    #train_data=data.iloc[:, 1:]
    train_target_values = train_data.values.T
    ##testing data

    test_data = data.iloc[:, 1:]

    test_target_values = test_data.values.T

    m = train_data.shape[1]
    print(m)
    print('starting_time')
    print(starting_time)
    # for start time :shape(list) len:m
    starting_dates = [pd.Timestamp(starting_time, freq=frequency) for _ in range(m)]
    print('starting_dates')
    print(starting_dates)
    # m5_dates = [pd.Timestamp("01-01-1990", freq='B') for _ in range(m)]
    # for category features: id shape(m,1)
    #stat_cat = np.array(range(m)).reshape(m, 1)
    # for time features: list, every element in the list is an array(shape:f*T) where f is the
    from gluonts.dataset.common import load_datasets, ListDataset
    from gluonts.dataset.field_names import FieldName
    train_ds = ListDataset([
        {
            FieldName.TARGET: target,
            FieldName.START: start
            # FieldName.FEAT_DYNAMIC_REAL: fdr,
            #FieldName.FEAT_STATIC_CAT: fsc
        }
        for (target, start) in zip(train_target_values,
                                        starting_dates)
    ], freq=frequency)
    print('train')

    print(next(iter(train_ds)))

    test_ds = ListDataset([
        {
            FieldName.TARGET: target,
            FieldName.START: start
            # FieldName.FEAT_DYNAMIC_REAL: fdr,
            #FieldName.FEAT_STATIC_CAT: fsc
        }
        for (target, start) in zip(test_target_values,
                                        starting_dates)
    ], freq=frequency)
    print('test')

    print(next(iter(test_ds)))
    estimator = DeepAREstimator(
        prediction_length=h,
        freq=frequency
    )

    predictor = estimator.train(train_ds)

    from gluonts.evaluation.backtest import make_evaluation_predictions

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor
    )
    print(type(forecast_it))
    print(type(ts_it))
    tss = list(tqdm(ts_it, total=len(test_ds)))
    print("Obtaining time series predictions ...")
    forecasts = list(tqdm(forecast_it, total=len(test_ds)))
    plot_forecasts(tss, forecasts, train_data.shape[0], m,figure_saving_path)
    # tss[0].head()
    point_forecasts_acc = np.zeros((len(forecasts), h))
    for i in range(len(forecasts)):
        point_forecasts_acc[i] = np.mean(forecasts[i].samples, axis=0)

    # point_forecasts_acc shape:m*horizen
    #
    lower_prediction_array=np.zeros((len(forecasts), h))
    upper_prediction_array = np.zeros((len(forecasts), h))
    for ts in range(m):
        # interval prediction
        # forecast_entry(100*8)
        forecast_entry = pd.DataFrame(forecasts[ts].samples)
        # begin to calculate quantile
        # lower_upper_df(2*48)
        lower_upper_df = forecast_entry.quantile([0.025, 0.975])
        print(lower_upper_df)
        lower_prediction_array[ts] = np.array(lower_upper_df.iloc[0, :].tolist())
        upper_prediction_array[ts] = np.array(lower_upper_df.iloc[1, :].tolist())
    return point_forecasts_acc,lower_prediction_array,upper_prediction_array






num_of_ts=3
horizons=6
num_of_forecast=20
#for yearly
frequency='A'
seasonality=1
level_value=95
name_of_dataset='temperature_data.csv'
all_data=pd.read_csv(name_of_dataset)
len_all=all_data.shape[0]
test_len=horizons+num_of_forecast-1
train_len=len_all-test_len

all_point_forecasts=np.zeros((num_of_ts,horizons,num_of_forecast))
all_lower_forecasts=np.zeros((num_of_ts,horizons,num_of_forecast))
all_upper_forecasts=np.zeros((num_of_ts,horizons,num_of_forecast))

save_path='./temp/deepAR/'
figure_path='./temp/deepAR/forecasts/'

#result_ids = []

for i in range(num_of_forecast):
    b=i
    e=i+train_len+horizons
    temp_train_data=all_data.iloc[b:e,:]
    print('e')
    print(e)
    starting_time=all_data.iloc[b,0]
    print('starting_time')
    print(starting_time)
    print(str(starting_time))
    figure_saving_path=figure_path+'num'+str(i)+'_'
    res=deepAR(temp_train_data,frequency,seasonality,horizons,str(starting_time),figure_saving_path)
    pd.DataFrame(res[0]).to_csv(save_path+'point/'+str(i)+'_'+'point_forecasts.csv')
    pd.DataFrame(res[1]).to_csv(save_path+'lower/'+str(i)+'_'+'lower_forecasts.csv')
    pd.DataFrame(res[2]).to_csv(save_path+'upper/'+str(i)+'_'+'upper_forecasts.csv')
    all_point_forecasts[:,:,i]=res[0]
    all_lower_forecasts[:,:,i]=res[1]
    all_upper_forecasts[:,:,i]=res[2]




#cal accuray

for ts in range(num_of_ts):
    mse_array=np.zeros((num_of_forecast,horizons))
    mape_array=np.zeros((num_of_forecast,horizons))
    msis_array=np.zeros((num_of_forecast,horizons))
    mis_array=np.zeros((num_of_forecast,horizons))
    ts_all=np.array(all_data.iloc[:,ts+1])
    for j in range(num_of_forecast):
        b=j
        e=j+train_len
        x=ts_all[b:e]
        xx=ts_all[e:(e+horizons)]
        predicted=all_point_forecasts[ts,:,j]
        mse_array[j,]=mse_cal(xx,predicted)
        mape_array[j,]=mape_cal(xx,predicted)
        msis_array[j,]=msis_cal(x,xx,all_upper_forecasts[ts,:,j],all_lower_forecasts[ts,:,j],0.05,seasonality,horizons)
    print('ts')
    print(ts)
    print('mse:h1-6')
    mse=np.mean(mse_array,axis=0)
    print(mse)
    print('h=1-3')
    print(np.mean(mse[0:3]))
    print('h=1-6')
    print(np.mean(mse))

    print('mape:h1-6')
    mape=np.mean(mape_array,axis=0)
    print(mape)
    print('h=1-3')
    print(np.mean(mape[0:3]))
    print('h=1-6')
    print(np.mean(mape))




    print('msis:h1-6')
    msis=np.mean(msis_array,axis=0)
    print(msis)
    print('h1-3')
    print(np.mean(msis[0:3]))
    print('h1-6')
    print(np.mean(msis))












