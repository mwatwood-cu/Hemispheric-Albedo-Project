import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def create_lat_weights():
    lat_weights = pd.read_csv("./Data/lat_weights.csv", header=0)
    lat_weights = lat_weights.to_numpy()

    lats = lat_weights[0:-1, 0]
    lats = lats.astype(float)
    vals = lat_weights[0:-1, 1]
    vals = vals.astype(float)

    cs = CubicSpline(lats, vals)
    xs = np.arange(-90, 90, 0.1)

    return vals

def calculate_deseasonalized_data(data):
    data_len = len(data)
    total_avg = data.mean()
    monthly_avgs = np.zeros((12, 1))
    for i in range(12):
        month_vals = data[range(i, data_len, 12)]
        monthly_avgs[i] = month_vals.mean()-total_avg
    
    avg_mon = data.copy()
    
    for i in range(data_len):
        avg_mon[i] = data[i]-monthly_avgs[i % 12][0]
    return avg_mon


def calculate_anomaly_from_monthly_avg(data):
    data_len = len(data)
    monthly_avgs = np.zeros((12, 1))
    for i in range(12):
        month_vals = data[range(i, data_len, 12)]
        monthly_avgs[i] = month_vals.mean()

    avg_mon = data.copy()

    for i in range(data_len):
        avg_mon[i] = data[i] - monthly_avgs[i % 12][0]
    return avg_mon

def create_lat_and_lon_weighted_mean(data, category, weights):
    specific_data = data[category].copy()
    weighted_lon = specific_data.weighted(weights)
    specific_data_weighted_mean = weighted_lon.mean(("lat", "lon"))

    return specific_data_weighted_mean

def create_monthly_sorted_data(data):
    num_years = round(len(data)/12)+1
    month_data = np.empty((num_years, 12))
    month_data[:] = np.NaN
    year_count = 0
    if data[0].time.dt.month==1:
        year_count = -1
    for i in range(len(data)):
        if (data[i].time.dt.month.data == 1):
            year_count = year_count + 1
        month_data[year_count, data[i].time.dt.month.data-1] = data[i].data

    # dt_index = pd.period_range("1/15/2000", periods=12, freq="M")
    dt_index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df = pd.DataFrame(month_data, columns=dt_index)
    df = remove_non_full_rows(df)
    return df

def remove_non_full_rows(df):
    num_rows = df.shape[0]   
    # Cleaning up and removing any rows that contain NaN
    for i in reversed(range(num_rows)):
        if df.iloc[i].isnull().any():
            df = df.drop(i)
    return df

def remove_non_full_cols(df):
    # Cleaning up and removing any rows that contain NaN
    for i in df.columns:
        if df[i].isnull().any():
            df = df.drop(columns=i)
    return df

def create_yearly_sorted_data(time_series_data):
    num_years = round(len(time_series_data)/12)
    if len(time_series_data)%12 != 0:
        num_years = num_years + 1
    year_data = np.empty((12, num_years))
    year_data[:] = np.NaN
    year_count = 0
    year_names = []
    if(time_series_data[0].time.dt.month.data != 1):
        year_names.append(str(time_series_data[0].time.dt.year.data))
    else:
        year_count = -1

    for i in range(len(time_series_data)):
        if (time_series_data[i].time.dt.month.data == 1):
            year_count = year_count + 1
            year_names.append(str(time_series_data[i].time.dt.year.data))
        year_data[time_series_data[i].time.dt.month.data-1, year_count] = time_series_data[i].data
    
    df = pd.DataFrame(year_data, columns=[year_names])
    df = remove_non_full_cols(df)
    return df

def make_datetime_once_per_year(start_year, end_year):
    date_strings = []
    for i in range(int(end_year)-int(start_year)+1):
        year = int(start_year) + i 
        date_string = str(year)+'-06-01'
        date_strings.append(date_string)

    dates = pd.DatetimeIndex(date_strings)
    return dates

def remove_incomplete_years(time_series):
    one_year_data = np.empty((12,1))
    one_year_data[:] = np.NaN
    start_index = len(time_series)-1
    for i in reversed(range(len(time_series))):
        current_month = int(time_series[i].time.dt.month) 
        one_year_data[current_month-1] = time_series[i]
        if(current_month == 1 or i == 0):
            if(np.isnan(one_year_data).any()):
                current_year = int(time_series[i].time.dt.year)
                time_series = time_series[time_series['time'].dt.year != current_year]
            start_index = i-1
            one_year_data = np.empty((12,1))
            one_year_data[:] = np.NaN
    return time_series

#def create_dataframe_with_month_and_year(date_series, variable_name):

