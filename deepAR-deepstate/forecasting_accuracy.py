import numpy as np
def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.abs(_error(actual, predicted))
def sme(training:np.ndarray,testing_actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return (_error(testing_actual, predicted))/np.mean(training)

def mase(training:np.ndarray,actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    T=len(training)
    return mae(actual, predicted) / np.mean( mae(training[seasonality:],training[0:(T-seasonality)] ))


def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return 200 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))

def msis_calc(upper_forecasts_array,lower_forecasts_array,real_value_array,a,freq,training_ts):
    #msis
    msis_array=np.empty([upper_forecasts_array.shape[0],upper_forecasts_array.shape[1]])
    #

    denominator_list=[]
    for m in range(upper_forecasts_array.shape[1]):
        ts_list=training_ts.iloc[:,(m+1)].tolist()
        one_ts_list=[]
        for t in range(freq,training_ts.shape[0]):
            one_ts_list.append(abs(ts_list[t]-ts_list[t-freq]))
        denominator_list.append(sum(one_ts_list)/len(one_ts_list))


    for h in range(upper_forecasts_array.shape[0]):
        for m in range(upper_forecasts_array.shape[1]):
            #calculate msis
            u=upper_forecasts_array[h,m]
            l=lower_forecasts_array[h,m]
            real=real_value_array[h,m]
            if real<l:
                p1=1
            else:
                p1=0
            if real>u:
                p2=1
            else:
                p2=0
            msis=(u-l)+(2/a)*(l-real)*p1+(2/a)*(real-u)*p2
            msis_array[h,m]=msis/denominator_list[m]

    return msis_array

def msis_cal(training,actual,upper,lower,a,seasonality,horizon):
    T = len(training)
    seasonaly_forecast_error=np.mean(mae(training[seasonality:],training[0:(T-seasonality)] ))
    interval=(upper-lower)
    lower_penalty=[]
    upper_penalty=[]
    for i in range(horizon):
        if actual[i]<lower[i]:
            lower_penalty.append((lower[i]-actual[i] )*(2/a))
        else:
            lower_penalty.append(0)
        if actual[i]>upper[i]:
            upper_penalty.append((actual[i]-upper[i])*(2/a))
        else:
            upper_penalty.append(0)
    lower_penalty_array=np.array(lower_penalty)
    upper_penalty_array=np.array(upper_penalty)
    return (interval+lower_penalty_array+upper_penalty_array)/(seasonaly_forecast_error)

def mse_cal(actual,predicted):
    return _error(actual,predicted)*_error(actual,predicted)

def mape_cal(actual,predicted):
    return np.abs(_error(actual,predicted))*100/(np.abs(actual))
#test


