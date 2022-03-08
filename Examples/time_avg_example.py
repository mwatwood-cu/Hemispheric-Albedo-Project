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

nh_solar_in = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'solar_mon', hemisphere="nh", deseason=False)
sh_solar_in = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'solar_mon', hemisphere="sh", deseason=False)
all_solar_in = create_weighted_mean_time_series_from_CERES_ERBF_data(ds, 'solar_mon', hemisphere="none", deseason=False)

nh_year_data = nh_solar_in.groupby("time.year").mean()
sh_year_data = sh_solar_in.groupby("time.year").mean()
all_year_data = all_solar_in.groupby("time.year").mean()

fig, ax = plt.subplots(2,1)

ax[0].plot(nh_year_data.year, nh_year_data, label="NH")
ax[0].plot(nh_year_data.year, sh_year_data, label="SH")
ax[0].set_xticks([])
ax[0].set_title("With No Temporal Weighting - Averaged Hemispheric Incoming Solar Flux Density")
ax[0].set_ylabel(r"W/m$^2$")

solar_data = ds["solar_mon"]
month_length = ds.time.dt.days_in_month
time_weights = time_weights = month_length.groupby("time.year")/month_length.groupby("time.year").sum()
solar_t_weighted = solar_data*time_weights
#solar_t_weighted = solar_t_weighted.groupby("time.year").sum()

lat_weights = create_lat_weights()
space_weights = xr.DataArray(data=lat_weights, coords=[solar_t_weighted.lat], dims=["lat"])
space_weights.name = "weights"  

nh_t = solar_t_weighted.sel(lat=slice(0, 90))
nh_t_w = nh_t.weighted(space_weights)
sh_t = solar_t_weighted.sel(lat=slice(-90, 0))
sh_t_w = sh_t.weighted(space_weights)
all_t = solar_t_weighted.weighted(space_weights)

nh_solar_ts = nh_t_w.mean(("lat","lon"))
sh_solar_ts = sh_t_w.mean(("lat","lon"))
all_solar_ts = all_t.mean(("lat","lon"))

nh_solar_ts = remove_incomplete_years(nh_solar_ts)
sh_solar_ts = remove_incomplete_years(sh_solar_ts)
all_solar_ts = remove_incomplete_years(all_solar_ts)

ax[1].plot(nh_year_data.year,nh_solar_ts.groupby("time.year").sum(), label="NH")
ax[1].plot(nh_year_data.year,sh_solar_ts.groupby("time.year").sum(), label="SH")
ax[1].plot(nh_year_data.year,all_solar_ts.groupby("time.year").sum(), 'k', label="Global Avg")
ax[0].plot(nh_year_data.year, all_solar_ts.groupby("time.year").sum(), 'k', label="Global Avg")
ax[1].set_xticks([2000,2004, 2008, 2012, 2016, 2020])
ax[1].set_ylabel(r"W/m$^2$")
ax[0].get_shared_x_axes().join(ax[0], ax[1])
ax[1].legend()
ax[1].set_title("With Temporal Weighting - Averaged Hemispheric Incoming Solar Flux Density")

a = 1