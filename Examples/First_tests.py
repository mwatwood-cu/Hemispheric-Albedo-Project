import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sys, os

curdir = os.getcwd()
sys.path.insert(0, curdir+"/Functions")

# My built function imports
from data_manipulation_functions import *
from essential_functions import *
from plotting_functions import *

years = "21"

file_name = "./Data/CERES_EBAF-TOA_Full.nc"

ds = xr.open_dataset(file_name)

# Making All-Sky Time Series
nh_all_sky = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'toa_sw_all_mon', hemisphere="nh")
sh_all_sky = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'toa_sw_all_mon', hemisphere="sh")
global_all_sky = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'toa_sw_all_mon', hemisphere="none")

nh_solar_in = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'solar_mon', hemisphere="nh")
sh_solar_in = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'solar_mon', hemisphere="sh")

# Making Clear-Sky Time Series
nh_clear_sky = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'toa_sw_clr_c_mon', hemisphere="nh")
sh_clear_sky = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'toa_sw_clr_c_mon', hemisphere="sh")

# Making Cloud Area Time Series
global_cloud_area = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'cldarea_total_daynight_mon', hemisphere="none")
nh_cloud_area = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'cldarea_total_daynight_mon', hemisphere="nh")
sh_cloud_area = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'cldarea_total_daynight_mon', hemisphere="sh")

# Calculating Differences
all_sky_diff = nh_all_sky - sh_all_sky
clear_sky_diff = nh_clear_sky - sh_clear_sky

as_monthly_data = create_monthly_sorted_data(all_sky_diff)
as_nh_month_data = create_monthly_sorted_data(nh_all_sky)
as_sh_month_data = create_monthly_sorted_data(sh_all_sky)

cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
#as_monthly_data.boxplot(column=cols)

#Plot all of the pieces togther
f, axs = plt.subplots(3, sharex=True)
plot_time_series_with_mean_std_once_per_year(all_sky_diff, title="Global All Sky Difference - "+years+" years", axes=axs[0])
plot_time_series_with_mean_std_once_per_year(nh_all_sky, title="NH All Sky Difference - "+years+" years", axes=axs[1])
plot_time_series_with_mean_std_once_per_year(sh_all_sky, title="SH All Sky Difference - "+years+" years", axes=axs[2])

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

plt.title("All Sky "+years+ " Year Monthly Box plot")
#plt.savefig("./Figs/"+years+"yearMonthlyBox.png")
plt.show()



