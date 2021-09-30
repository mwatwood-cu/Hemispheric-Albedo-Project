import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# My built function imports
from data_manipulation_functions import *
from essential_functions import *
from plotting_functions import *


file_name = "CERES_EBAF-TOA_10years.nc"

ds = xr.open_dataset(file_name)

# Making All-Sky Time Series
nh_all_sky = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'toa_sw_all_mon', hemisphere="nh")
sh_all_sky = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'toa_sw_all_mon', hemisphere="sh")

# Making Clear-Sky Time Series
nh_clear_sky = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'toa_sw_clr_c_mon', hemisphere="nh")
sh_clear_sky = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'toa_sw_clr_c_mon', hemisphere="sh")

# Calculating Differences
all_sky_diff = nh_all_sky - sh_all_sky
clear_sky_diff = nh_clear_sky - sh_clear_sky

as_monthly_data = create_monthly_sorted_data(all_sky_diff)
cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
as_monthly_data.boxplot(column=cols)

plot_time_series_with_mean_std_once_per_year(all_sky_diff)
plot_time_series_with_mean_std_once_per_year(nh_all_sky)
plot_time_series_with_mean_std_once_per_year(sh_all_sky)

print("Mean NH SW all sky is " + str(nh_all_sky.mean().data))
print("Mean SH SW all sky is " + str(sh_all_sky.mean().data))
print("Mean NH SW clear sky is " + str(nh_clear_sky.mean().data))
print("Mean SH SW clear sky is " + str(sh_clear_sky.mean().data))

print("Average all sky hemispheric symmetry difference is " + str(all_sky_diff.mean().data))
print("Average clear sky hemispheric symmetry difference is " + str(clear_sky_diff.mean().data))

# Plot full Data
#all_sky_diff.plot()

# Plot Running Avg - time = months of rolling avg
#running_all_sky = all_sky_diff.rolling(time=6).mean()
#running_all_sky.plot()

plt.title("All Sky 10 Year Monthly Box plot")
#plt.savefig("./Figs/10yearMonthlyBox.png")
plt.show()



