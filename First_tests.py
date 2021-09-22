import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from lat_weights import create_lat_weights


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


def create_hemisphere_mean(data, category, weights):
    specific_data = data[category].copy()
    weighted_lon = specific_data.weighted(weights)
    specific_data_weighted_mean = weighted_lon.mean(("lat", "lon"))

    return specific_data_weighted_mean


def create_monthly_sorted_data(data):
    num_years = round(len(data)/12)+1
    month_data = np.empty((num_years, 12))
    month_data[:] = np.NaN
    year_count = 0
    for i in range(len(data)):
        if (data[i].time.dt.month.data == 1):
            year_count = year_count + 1
        month_data[year_count, data[i].time.dt.month.data-1] = data[i].data
    df = pd.DataFrame(month_data, columns={'Jan', 'Feb',
                                           'Mar', 'Apr',
                                           'May', 'Jun',
                                           'Jul', 'Aug',
                                           'Sep', 'Oct',
                                           'Nov', 'Dec'})
    # Cleaning up and removing any rows that contain NaN
    for i in range(num_years):
        if df.iloc[i].isnull().any():
            df.drop(i)

    return df

file_name = "CERES_EBAF-TOA_21years.nc"

ds = xr.open_dataset(file_name)

lat_weights = create_lat_weights()
weights = xr.DataArray(data=lat_weights, coords=[ds.lat], dims=["lat"])
weights.name = "weights"

nh = ds.sel(lat=slice(0, 90))
nh_weights = weights.sel(lat=slice(0, 90))
sh = ds.sel(lat=slice(-90, 0))
sh_weights = weights.sel(lat=slice(-90, 0))

nh_all_sky_mean = create_hemisphere_mean(nh, 'toa_sw_all_mon', nh_weights)
nh_all_sky_mean_year = calculate_deseasonalized_data(nh_all_sky_mean)
sh_all_sky_mean = create_hemisphere_mean(sh, 'toa_sw_all_mon', sh_weights)
sh_all_sky_mean_year = calculate_deseasonalized_data(sh_all_sky_mean)

nh_clear_sky_mean = create_hemisphere_mean(nh, 'toa_sw_clr_c_mon', nh_weights)
nh_clear_sky_mean_year = calculate_deseasonalized_data(nh_clear_sky_mean)
sh_clear_sky_mean = create_hemisphere_mean(sh, 'toa_sw_clr_c_mon', sh_weights)
sh_clear_sky_mean_year = calculate_deseasonalized_data(sh_clear_sky_mean)

all_sky_diff = nh_all_sky_mean_year - sh_all_sky_mean_year
clear_sky_diff = nh_clear_sky_mean_year - sh_clear_sky_mean_year

as_monthly_data = create_monthly_sorted_data(all_sky_diff)

cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
as_monthly_data.boxplot(column=cols)

print("Mean NH SW all sky is " + str(nh_all_sky_mean_year.mean().data))
print("Mean SH SW all sky is " + str(sh_all_sky_mean_year.mean().data))
print("Mean NH SW clear sky is " + str(nh_clear_sky_mean_year.mean().data))
print("Mean SH SW clear sky is " + str(sh_clear_sky_mean_year.mean().data))

print("Average all sky hemispheric symmetry difference is " + str(all_sky_diff.mean().data))
print("Average clear sky hemispheric symmetry difference is " + str(clear_sky_diff.mean().data))

# Plot full Data
#all_sky_diff.plot()

# Plot Running Avg - time = months of rolling avg
#running_all_sky = all_sky_diff.rolling(time=6).mean()
#running_all_sky.plot()

plt.title("All Sky 21 Year Monthly Box plot")
plt.savefig("./Figs/21yearMonthlyBox.png")
plt.show()



