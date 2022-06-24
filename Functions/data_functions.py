from regex import D
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def pick_CERES_data(dataset, start_yr, start_mon, end_yr, end_mon):
    start = start_yr+"-"+start_mon
    end = end_yr+"-"+end_mon
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
    kernels[2,0] = -0.50
    # Add 3/4 of a day from year 4
    kernels[2,12] = 0.75
    # Remove that from year 4
    kernels[3,0] = -0.75

    return kernels


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


def apply_time_averaging(dataset, leap_year_method=0, feb_leap_year_correction=27.65, 
    feb_non_leap_year_correction=28.45, running_length=0, start_mon="01"):

    month_length = dataset.time.dt.days_in_month

    # When not using a leap year correction
    if(leap_year_method == 0):
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
    if(leap_year_method == 1):
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

    #if(leap_year_method == 2):
    # Here we do use the leap year correction of adjusting first and last month
    time_weights = month_length.values.astype("float")
    addition_kernels = make_leap_year_addition_kernels()
    years = np.unique(month_length.time.values.astype('datetime64[Y]').astype("float") + 1970)
    years = years[:-1]

    new_data = dataset.groupby("time.year").sum().copy()
    # Here there is no running average
    if(running_length == 0):
        for count, year in enumerate(years):
            if(start_mon == "01" or "02"):
                leap_year_offset = int((year-1)%4)
            else:
                leap_year_offset = int(year%4)
            days = time_weights[count*12:(count+1)*12+1].copy()
            data = dataset[count*12:(count+1)*12+1]
            addition = addition_kernels[leap_year_offset]
            
            if(len(days)== 13):
                days[12] = 0
            if(len(days)== 12):
                addition = addition[:-1]

            days = days+addition
            weights = days/days.sum()
            if(len(dataset.shape)==3):
                yearly_avg = np.average(data, weights=weights, axis=0)
                new_data[count,:,:] = yearly_avg
            elif(len(dataset.shape)==1):
                yearly_avg = np.average(data, weights=weights, axis=0)
                new_data[count] = yearly_avg
            
        return new_data[:-1]

    #if(running_length != 0):
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


def apply_spatial_weights(dataset, low_lat=0, high_lat=90):
    lat_weights = create_lat_weights()

    space_weights = xr.DataArray(data=lat_weights, coords=[dataset.lat], dims=["lat"])
    space_weights.name = "weights"  

    slice_of_data = dataset.sel(lat=slice(low_lat, high_lat))
    weighted_slice = slice_of_data.weighted(space_weights)

    mean_slice = weighted_slice.mean(("lat","lon"))
    return mean_slice


def create_CERES_hemisphere_data(dataset, category, use_time_weighting=True, use_spatial_weighting=True,
    remove_leap_year=1, running_length=0, start_yr="2001", start_mon="01", end_yr="2022", end_mon="01"):
    specific_dat = dataset[category]
    
    cleaned_dat = pick_CERES_data(specific_dat, start_yr, start_mon, end_yr, end_mon)

    specific_t_weighted = apply_time_averaging(cleaned_dat, leap_year_method=remove_leap_year, 
        running_length=running_length, start_mon=start_mon)

    nh_ts_mean = apply_spatial_weights(specific_t_weighted, low_lat=0, high_lat=90)
    sh_ts_mean = apply_spatial_weights(specific_t_weighted, low_lat=-90, high_lat=0)
    all_ts_mean = apply_spatial_weights(specific_t_weighted, low_lat=-90, high_lat=90)

    ds_new = xr.Dataset({"nh":nh_ts_mean, "sh":sh_ts_mean, "global":all_ts_mean})
    return ds_new

def create_CERES_20_degree_zonal_data(dataset, category,
    remove_leap_year=1, running_length=0, start_yr="2001", start_mon="01", end_yr="2022", end_mon="01"):
    specific_dat = dataset[category]

    cleaned_dat = pick_CERES_data(specific_dat, start_yr, start_mon, end_yr, end_mon)

    specific_t_weighted = apply_time_averaging(cleaned_dat, leap_year_method=remove_leap_year, 
        running_length=running_length, start_mon=start_mon)

    neg_80 = apply_spatial_weights(specific_t_weighted, low_lat=-90, high_lat=-70)
    neg_60 = apply_spatial_weights(specific_t_weighted, low_lat=-70, high_lat=-50)
    neg_40 = apply_spatial_weights(specific_t_weighted, low_lat=-50, high_lat=-30)
    neg_20 = apply_spatial_weights(specific_t_weighted, low_lat=-30, high_lat=-10)
    zero_band = apply_spatial_weights(specific_t_weighted, low_lat=-10, high_lat=10)
    pos_20 = apply_spatial_weights(specific_t_weighted, low_lat=10, high_lat=30)
    pos_40 = apply_spatial_weights(specific_t_weighted, low_lat=30, high_lat=50)
    pos_60 = apply_spatial_weights(specific_t_weighted, low_lat=50, high_lat=70)
    pos_80 = apply_spatial_weights(specific_t_weighted, low_lat=70, high_lat=90)

    ds_new = xr.Dataset({"neg_80":neg_80, "neg_60":neg_60, "neg_40":neg_40, "neg_20":neg_20,
                        "zero":zero_band,
                        "pos_80":pos_80, "pos_60":pos_60, "pos_40":pos_40, "pos_20":pos_20,}) 
    return ds_new

def create_CERES_30_degree_zonal_data(dataset, category,
    remove_leap_year=1, running_length=0, start_yr="2001", start_mon="01", end_yr="2022", end_mon="01"):
    specific_dat = dataset[category]

    cleaned_dat = pick_CERES_data(specific_dat, start_yr, start_mon, end_yr, end_mon)

    specific_t_weighted = apply_time_averaging(cleaned_dat, leap_year_method=remove_leap_year, 
        running_length=running_length, start_mon=start_mon)

    neg_75 = apply_spatial_weights(specific_t_weighted, low_lat=-90, high_lat=-60)
    neg_45 = apply_spatial_weights(specific_t_weighted, low_lat=-60, high_lat=-30)
    neg_15 = apply_spatial_weights(specific_t_weighted, low_lat=-30, high_lat=0)

    pos_15 = apply_spatial_weights(specific_t_weighted, low_lat=0, high_lat=30)
    pos_45 = apply_spatial_weights(specific_t_weighted, low_lat=30, high_lat=60)
    pos_75 = apply_spatial_weights(specific_t_weighted, low_lat=60, high_lat=90)

    ds_new = xr.Dataset({"neg_75":neg_75, "neg_45":neg_45, "neg_15":neg_15,
                        "pos_75":pos_75, "pos_45":pos_45, "pos_15":pos_15}) 
    return ds_new