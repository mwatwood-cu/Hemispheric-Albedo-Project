import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sys, os

curdir = os.getcwd()
sys.path.insert(0, curdir+"/Functions")

# My built function imports
from data_functions import *
from plotting_functions import *

file_name = "./Data/CERES_EBAF-TOA_Full.nc"
ds = xr.open_dataset(file_name)

full_years = clean_CERES_data(ds)

solar_in = create_CERES_hemisphere_data(full_years, "solar_mon")

sw_all = create_CERES_hemisphere_data(full_years, "toa_sw_all_mon")
sw_clear = create_CERES_hemisphere_data(full_years, "toa_sw_clr_c_mon")

lw_all = create_CERES_hemisphere_data(full_years, "toa_lw_all_mon")
lw_clear = create_CERES_hemisphere_data(full_years, "toa_lw_clr_c_mon")

net_all = create_CERES_hemisphere_data(full_years, "toa_net_all_mon")
net_clear = create_CERES_hemisphere_data(full_years, "toa_net_clr_c_mon")

cloud_area = create_CERES_hemisphere_data(full_years, "cldarea_total_daynight_mon")
cloud_pressure = create_CERES_hemisphere_data(full_years, "cldpress_total_daynight_mon")
cloud_temp = create_CERES_hemisphere_data(full_years, "cldtemp_total_daynight_mon")
cloud_tau = create_CERES_hemisphere_data(full_years, "cldtau_total_day_mon")

plot_CERES_sw_lw_net(sw_all, sw_clear, lw_all, lw_clear, net_all, net_clear)
plot_CERES_asymmetry(sw_all, sw_clear, lw_all, lw_clear, net_all, net_clear, solar_in, 
    cloud_area, cloud_pressure, cloud_temp, cloud_tau)


a = 8