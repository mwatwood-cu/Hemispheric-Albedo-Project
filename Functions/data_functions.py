from regex import D
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

LEAP_YEAR_CORRECTION = (366/1461)/0.25
NON_LEAP_YEAR_CORRECTION = (365/1461)/0.25

def clean_CERES_data(dataset, start, end):
    data = dataset.sel(time=slice(start, end))
    return data

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

def make_leap_year_addition_kernels():
    kernels = np.zeros((4,13))
    # First year gets 1/4 day from next year
    kernels[0,12] = 0.25
    # Remove that from second year
    kernels[1,0] = -0.25
    # Add 1/2 day from year 3
    kernels[1,12] = 0.50
    # Remove that from year 3
    kernels[2,1] = -0.50
    # Add 3/4 of a day from year 4
    kernels[2,12] = 0.75
    # Remove that from year 4
    kernels[3,0] = -0.75

    return kernels

def apply_time_averaging(dataset, use_leap_year_adjust=True, feb_leap_year_correction=27.65, 
    feb_non_leap_year_correction=28.45, running_length=0):

    month_length = dataset.time.dt.days_in_month
    '''
    if(use_leap_year_adjust):
        month_length.values = month_length.values.astype("float")
        month_length[month_length==28] = feb_non_leap_year_correction
        month_length[month_length==29] = feb_leap_year_correction
    time_weights = month_length.groupby("time.year")/month_length.groupby("time.year").sum()'''

    time_weights = month_length.values.astype("float")
    addition_kernels = make_leap_year_addition_kernels()
    

    if(running_length != 0):
        if(len(dataset.shape)==3):
            # Create the running avg data
            t_weighted = np.array([np.average(dataset[x:x+running_length],  \
                weights=time_weights[x:x+running_length], axis=0)for x in range(len(dataset)-running_length)])
        elif(len(dataset.shape)==1):
            t_weighted =np.array([np.average(dataset[x:x+running_length],  \
                weights=time_weights[x:x+running_length])for x in range(len(dataset)-running_length)]) 
        # Shorten the original data to match the running avg length
        specific_t_weighted = dataset[:len(dataset)-running_length,:,:]
        specific_t_weighted.data = t_weighted
    else:
        # Make a yearly avg
        specific_t_weighted = dataset*time_weights
        specific_t_weighted = specific_t_weighted.groupby("time.year").sum()

    return specific_t_weighted

def apply_spatial_weights(dataset, low_lat=0, high_lat=90):
    lat_weights = create_lat_weights()

    space_weights = xr.DataArray(data=lat_weights, coords=[dataset.lat], dims=["lat"])
    space_weights.name = "weights"  

    slice_of_data = dataset.sel(lat=slice(low_lat, high_lat))
    weighted_slice = slice_of_data.weighted(space_weights)

    mean_slice = weighted_slice.mean(("lat","lon"))
    return mean_slice

def create_CERES_hemisphere_data(dataset, category, use_time_weighting=True, use_spatial_weighting=True,
    remove_leap_year=True, leap_year_correction=27.65, non_leap_year_correction=28.45, running_avg=False, 
    running_length=12, start="jan"):
    specific_dat = dataset[category]
    
    if(start=="mar"):
        dataset = clean_CERES_data(dataset, "2000-03", "2021-02")
    else:
        dataset = clean_CERES_data(dataset, "2001-01", "2021-12")

    specific_t_weighted = apply_time_averaging(specific_dat, use_leap_year_adjust=remove_leap_year, 
        feb_leap_year_correction=leap_year_correction, feb_non_leap_year_correction=non_leap_year_correction,
        use_running_avg=running_avg, running_length=running_length)

    nh_ts_mean = apply_spatial_weights(specific_t_weighted, low_lat=0, high_lat=90)
    sh_ts_mean = apply_spatial_weights(specific_t_weighted, low_lat=-90, high_lat=0)
    all_ts_mean = apply_spatial_weights(specific_t_weighted, low_lat=-90, high_lat=90)

    ds_new = xr.Dataset({"nh":nh_ts_mean, "sh":sh_ts_mean, "global":all_ts_mean})
    return ds_new

def create_yearly_sorted_data(time_series_data):
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