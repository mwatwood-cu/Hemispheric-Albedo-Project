import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def clean_CERES_data(dataset):
    years = dataset.time.dt.year
    month_count = years.groupby("time.year").count()
    #for dat in month_count.data:
        #if(dat)<12:
            #Remove?
            # 
    
    #Return a modified dataset
    return dataset  

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

def create_CERES_hemisphere_data(dataset, category):
    specific_dat = dataset[category]
    month_length = dataset.time.dt.days_in_month
    time_weights = time_weights = month_length.groupby("time.year")/month_length.groupby("time.year").sum()
    specific_t_weighted = specific_dat*time_weights

    lat_weights = create_lat_weights()
    space_weights = xr.DataArray(data=lat_weights, coords=[specific_t_weighted.lat], dims=["lat"])
    space_weights.name = "weights"  

    nh_t = specific_t_weighted.sel(lat=slice(0, 90))
    nh_t_w = nh_t.weighted(space_weights)
    sh_t = specific_t_weighted.sel(lat=slice(-90, 0))
    sh_t_w = sh_t.weighted(space_weights)
    all_t_w = specific_t_weighted.weighted(space_weights)

    nh_ts_mean = nh_t_w.mean(("lat","lon"))
    sh_ts_mean = sh_t_w.mean(("lat","lon"))
    all_ts_mean = all_t_w.mean(("lat","lon"))

    nh_yearly = nh_ts_mean.groupby("time.year").sum()
    sh_yearly = sh_ts_mean.groupby("time.year").sum()
    all_yearly = all_ts_mean.groupby("time.year").sum()

    ds_new = xr.Dataset({"nh":nh_yearly, "sh":sh_yearly, "global":all_yearly})
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