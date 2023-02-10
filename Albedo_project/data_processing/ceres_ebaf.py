import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import os

DATA_PATH = "/Users/mawa7160/dev/data/CERES/"

def slice_dataset_with_year_and_month(ceres_dataset: xr.Dataset, start_yr: str, start_mon, end_yr, end_mon):
    """
    Wrapper function to help select by month and year on a Pandas DataFrame
    """
    start = start_yr+"-"+start_mon
    end = end_yr+"-"+end_mon
    data = ceres_dataset.sel(time=slice(start, end))
    return data


def create_lat_weights(method: int, data_path: str=DATA_PATH, degree_step=1):
    if(method == 1):
        lat_weights = pd.read_csv(data_path+"lat_weights.csv", header=0)
        lat_weights = lat_weights.to_numpy()

        lats = lat_weights[0:-1, 0]
        lats = lats.astype(float)
        vals = lat_weights[0:-1, 1]
        vals = vals.astype(float)
    elif(method == 2):
        lats = np.arange(-90,90, 1)
        cosine_vals = np.cos(lats/180*np.pi)
        normalized_cosine = cosine_vals/np.sum(cosine_vals)
        vals = normalized_cosine
    
    if(degree_step == 1):
        return vals
    else:
        cs = CubicSpline(lats, vals)
        xs = np.arange(-90, 90, 0.1)
        return vals


def apply_spatial_weights(dataset, low_lat=0, high_lat=90, method=1):
    lat_weights = create_lat_weights(method)

    space_weights = xr.DataArray(data=lat_weights, coords=[dataset.lat], dims=["lat"])
    space_weights.name = "weights"  

    slice_of_data = dataset.sel(lat=slice(low_lat, high_lat))
    weighted_slice = slice_of_data.weighted(space_weights)

    mean_slice = weighted_slice.mean(("lat","lon"))
    return mean_slice


def calculate_weighted_mean(dataset, day_in_months):

    specific_t_weighted = dataset.groupby("time.year").sum().copy()

    time_weights = np.zeros(len(dataset))
    year_count = int(np.floor(len(dataset)/12))
    difference = len(specific_t_weighted)-year_count
    if(difference>0):
        specific_t_weighted = specific_t_weighted[:-difference]
    for i in range(year_count):
        time_weights[i*12:(i+1)*12] = day_in_months[i*12:(i+1)*12]/np.sum(day_in_months[i*12:(i+1)*12])
        if(len(dataset.shape)==3):
            specific_t_weighted[i] = np.average( dataset[i*12:(i+1)*12,:,:], 
                weights=time_weights[i*12:(i+1)*12], axis=0 )
        else:
            specific_t_weighted[i] = np.average( dataset[i*12:(i+1)*12,:,:], 
                weights=time_weights[i*12:(i+1)*12])

    return specific_t_weighted


def apply_time_averaging(dataset, averaging_method=0, feb_leap_year_correction=27.65, 
                         feb_non_leap_year_correction=28.45, running_length=0, start_mon="01"):

    month_length = dataset.time.dt.days_in_month

    # Incorrectly calculate with equal weight for each month
    if(averaging_method == -1):
        month_length_equal = np.ones_like(month_length)
        specific_t_weighted = calculate_weighted_mean(dataset, month_length_equal)
        return specific_t_weighted

    # When not using a leap year correction
    if(averaging_method == 0):
        # First case no running average
        if(running_length == 0):
            specific_t_weighted = calculate_weighted_mean(dataset, month_length)

        # Otherwise there is a running average
        else:
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
        return specific_t_weighted

    # Use the February only correction
    if(averaging_method == 1):
        month_length.values = month_length.values.astype("float")
        month_length[month_length==28] = feb_non_leap_year_correction
        month_length[month_length==29] = feb_leap_year_correction

        specific_t_weighted = calculate_weighted_mean(dataset, month_length)

        if(running_length != 0):
            # TODO Fix this to match new ideas.
            if(len(dataset.shape)==3):
                # Create the running avg data
                t_weighted = np.array([np.average(dataset[x:x+running_length],  \
                    weights=time_weights[x:x+running_length], axis=0)for x in range(len(dataset)-running_length)])
                # Shorten the original data to match the running avg length
                specific_t_weighted = dataset[:len(dataset)-running_length,:,:]
            elif(len(dataset.shape)==1):
                t_weighted =np.array([np.average(dataset[x:x+running_length],  \
                    weights=time_weights[x:x+running_length])for x in range(len(dataset)-running_length)]) 
                # Shorten the original data to match the running avg length
                specific_t_weighted = dataset[:len(dataset)-running_length]
            
            specific_t_weighted.data = t_weighted

        return specific_t_weighted
    
    if(averaging_method == 2):
        month_length.values = month_length.values.astype("float")
        year_count = int(np.floor(len(dataset)/12))
        for i in range(year_count):
            year_months = month_length[12*i:12*(i+1)]
            if(year_months.where(year_months.isin(29), drop=True).size == 1):
                month_length.values[i*12] = month_length.values[i*12]-3/8
                month_length.values[(i+1)*12-1] = month_length.values[(i+1)*12-1]-3/8
            else:
                month_length.values[i*12] = month_length.values[i*12]+1/8
                month_length.values[(i+1)*12-1] = month_length.values[(i+1)*12-1]+1/8

        specific_t_weighted = calculate_weighted_mean(dataset, month_length)
        return specific_t_weighted


def create_hemisphere_data(dataset,time_weighting=1, space_weighting=1,start_yr="2001", 
                                 start_mon="01", end_yr="2022", end_mon="01", ly_feb=27.65, nly_feb=28.45):
    
    cleaned_dat = slice_dataset_with_year_and_month(dataset, start_yr, start_mon, end_yr, end_mon)

    specific_t_weighted = apply_time_averaging(cleaned_dat, averaging_method=time_weighting, 
        running_length=0, start_mon=start_mon, feb_leap_year_correction=ly_feb, feb_non_leap_year_correction=nly_feb)

    nh_ts_mean = apply_spatial_weights(specific_t_weighted, low_lat=0, high_lat=90, method=space_weighting)
    sh_ts_mean = apply_spatial_weights(specific_t_weighted, low_lat=-90, high_lat=0, method=space_weighting)
    all_ts_mean = apply_spatial_weights(specific_t_weighted, low_lat=-90, high_lat=90, method=space_weighting)

    ds_new = xr.Dataset({"nh":nh_ts_mean, "sh":sh_ts_mean, "global":all_ts_mean})
    return ds_new