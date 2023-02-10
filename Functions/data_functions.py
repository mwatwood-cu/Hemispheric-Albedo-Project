import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

data_path = "/Users/mawa7160/dev/data/CERES/"

def pick_CERES_data(dataset, start_yr, start_mon, end_yr, end_mon):
    start = start_yr+"-"+start_mon
    end = end_yr+"-"+end_mon
    data = dataset.sel(time=slice(start, end))
    return data


def create_lat_weights():
    lat_weights = pd.read_csv(data_path+"lat_weights.csv", header=0)
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

    # Incorrectly calculate with equal weight for each month
    if(leap_year_method == -1):
        month_length_equal = np.ones_like(month_length)
        specific_t_weighted = calculate_weighted_mean(dataset, month_length_equal)
        return specific_t_weighted

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

def take_yearly_average_from_daily_data(dataset, start_year, end_year):
    num_years = int(end_year)-int(start_year)+1

    four_year_weights = np.ones((366,4))
    four_year_weights[0,0] = 0.25 # Account for missing from years 1,2, and 3

    four_year_weights[0,1] = 1 # Use the full first day of year
    four_year_weights[-1,1] = 0.25 # First year take 1/4 day from second year
    four_year_weights[0,2] = 0.75 # Account for missing 1/4 in year 2
    four_year_weights[-1,2] = 0.5 # Take 1/4 day from year 3
    four_year_weights[0,3] = 0.5 # Account for missing 1/2 day from year 1 and 2
    four_year_weights[-1,3] = 0.75 # Take 1/4 day from year 4

    nh = []
    sh = []
    glob = []
    years = []
    for year_index in range(num_years):
        this_year_str = str(int(start_year)+year_index)
        next_year_str = str(int(start_year)+year_index+1)

        leap_year_index = (int(start_year)+year_index)%4 # 0 means it is a leap year
        if(leap_year_index != 0):
            this_year = dataset.sel(time=slice(this_year_str+"-01-01",next_year_str+"-01-01"))
        else:
            this_year = dataset.sel(time=slice(this_year_str+"-01-01", this_year_str+"-12-31"))
        nh_dat = apply_spatial_weights(this_year, low_lat=0, high_lat=90)
        sh_dat = apply_spatial_weights(this_year, low_lat=-90, high_lat=0)
        global_dat = apply_spatial_weights(this_year, low_lat=-90, high_lat=90)

        nh_dat = np.average(nh_dat, weights=four_year_weights[:,leap_year_index], axis=0)
        sh_dat = np.average(sh_dat, weights=four_year_weights[:,leap_year_index], axis=0)
        global_dat = np.average(global_dat, weights=four_year_weights[:,leap_year_index], axis=0)

        years.append(int(start_year)+year_index)
        nh.append(nh_dat)
        sh.append(sh_dat)
        glob.append(global_dat)

    return xr.Dataset({"nh":nh, "sh":sh,"global":glob,}, coords={"years":years})

def create_CERES_EBAF_zonal_data(dataset, category, zone_size,
    remove_leap_year=1, running_length=0, start_yr="2001", start_mon="01", end_yr="2022", end_mon="01"):
    specific_dat = dataset[category]

    cleaned_dat = pick_CERES_data(specific_dat, start_yr, start_mon, end_yr, end_mon)

    specific_t_weighted = apply_time_averaging(cleaned_dat, leap_year_method=remove_leap_year, 
        running_length=running_length, start_mon=start_mon)

    num_zones = int(180/zone_size)
    num_years = len(specific_t_weighted.year)

    raw_data = np.ndarray((num_years, num_zones))
    lats = np.ndarray((num_zones))
    for i in range(num_zones):
        zone_start = -90+i*zone_size
        lats[i] = zone_start+zone_size/2
        raw_data[:,i]  = apply_spatial_weights(specific_t_weighted, low_lat=zone_start, high_lat=zone_start+zone_size)

    da = xr.DataArray(raw_data, coords=[specific_t_weighted.year, lats], dims=["year", "lat"])
    da.attrs["zonal_size"] = zone_size

    return da

def make_zonal_dataset_name(zone_start, zone_size):
    if(zone_start<0 and zone_start+zone_size<=0):
        zone_name = "neg_"+str(np.abs(np.round(zone_start+zone_size/2,1)))
    else:
        zone_name = str(np.abs(np.round(zone_start+zone_size/2,1)))

    if(zone_name[-1]=='0'):
        zone_name = zone_name[:-2]
    else:
        zone_name = zone_name.replace(".","_")

    return zone_name