# Installed
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sys, os
from sklearn.metrics import mean_squared_error

# Local Imports
curdir = os.getcwd()
sys.path.insert(0, curdir+"/Functions")
# My built function imports
from data_functions import *

current_month_folder="June22"

from plotting_functions import *

file_name = "./Data/CERES_EBAF-TOA_Full_2022_01.nc"
ds = xr.open_dataset(file_name)

full_years = ds

solar_in = create_CERES_hemisphere_data(full_years, "solar_mon", start_yr="2001", start_mon="01", 
    end_yr="2022", end_mon="01", remove_leap_year=1)

shortwave_all_sky = create_CERES_hemisphere_data(full_years, "toa_sw_all_mon", start_yr="2001", start_mon="01", 
    end_yr="2022", end_mon="01", remove_leap_year=1)

shortwave_clear_sky = create_CERES_hemisphere_data(full_years, "toa_sw_clr_c_mon", start_yr="2001", start_mon="01", 
    end_yr="2022", end_mon="01", remove_leap_year=1)

cre_total = shortwave_clear_sky-shortwave_all_sky

cre_diff = cre_total["nh"]-cre_total["sh"]
clear_sky_diff = shortwave_clear_sky["nh"]-shortwave_clear_sky["sh"]
all_sky_diff = shortwave_all_sky["nh"]-shortwave_all_sky["sh"]


# Plotting the all sky difference in SW
plot_data_with_reg_line(all_sky_diff.year, all_sky_diff, title="Hemispheric All Sky Albedo Difference (NH-SH)", 
                        include_trends=True, y_label="Albedo Difference W/m^2")
#plt.savefig("./Figs/"+current_month_folder+"/All-Sky-Difference")
plt.close()

# Plotting the clear sky difference in SW
plot_data_with_reg_line(clear_sky_diff.year, clear_sky_diff, title="Hemispheric Clear Sky Albedo Difference (NH-SH)", 
                        include_trends=True, y_label="Albedo Difference W/m^2") 
#plt.savefig("./Figs/"+current_month_folder+"/Clear-Sky-Difference")
plt.close()

# Plotting the CRE difference in SW
plot_data_with_reg_line(cre_diff.year, cre_diff, title="Hemispheric CRE (Clear-All) Albedo Difference (NH-SH)", 
                        include_trends=True, y_label="Albedo Difference W/m^2") 
#plt.savefig("./Figs/"+current_month_folder+"/CRE-Difference")
plt.close()

# Plotting the All sky data by hemisphere with trends
plot_data_with_reg_line(shortwave_all_sky.year, shortwave_all_sky["nh"], title="Hemispheric All Sky Albedo", 
                        include_trends=True, y_label="Albedo Difference W/m^2", label="NH", tick_color="g") 
plot_data_with_reg_line(shortwave_all_sky.year, shortwave_all_sky["sh"], title="Hemispheric All Sky Albedo", 
                        include_trends=True, y_label="Albedo Difference W/m^2", label="SH", tick_color="b")
#plt.savefig("./Figs/"+current_month_folder+"/All-Sky-by-Hemisphere")
plt.close()

# Plotting the clear sky data by hemisphere with trends
plot_data_with_reg_line(shortwave_clear_sky.year, shortwave_clear_sky["nh"], title="Hemispheric Clear Sky Albedo", 
                        include_trends=True, y_label="Albedo Difference W/m^2", label="NH", tick_color="g") 
plot_data_with_reg_line(shortwave_clear_sky.year, shortwave_clear_sky["sh"], title="Hemispheric Clear Sky Albedo", 
                        include_trends=True, y_label="Albedo Difference W/m^2", label="SH", tick_color="b") 
#plt.savefig("./Figs/"+current_month_folder+"/Clear-Sky-by-Hemisphere")
plt.close()


# Plotting the CRE data by hemisphere with trends
plot_data_with_reg_line(cre_total.year, cre_total["nh"], title="Hemispheric CRE Albedo", 
                        include_trends=True, y_label="Albedo Difference W/m^2", label="NH", tick_color="g") 
plot_data_with_reg_line(cre_total.year, cre_total["sh"], title="Hemispheric CRE Albedo", 
                        include_trends=True, y_label="Albedo Difference W/m^2", label="SH", tick_color="b") 
#plt.savefig("./Figs/"+current_month_folder+"/CRE-by-Hemisphere")
plt.close()

# Plotting the basics!
fig, ax = plt.subplots(1,4, figsize=(12,6))
plot_hemisphere_and_global_by_year(solar_in, solar_in.year, axis = ax[0], set_short_x_ticks=True, include_trends=True)
plot_hemisphere_and_global_by_year(shortwave_all_sky, shortwave_all_sky.year, title="All Sky Shortwave", 
    fixed_ylim=False, y_label="Flux W/m^2", axis=ax[1], set_short_x_ticks=True, include_trends=True)
plot_hemisphere_and_global_by_year(shortwave_clear_sky, shortwave_clear_sky.year, title="Clear Sky Shortwave", 
    fixed_ylim=False, y_label="Flux W/m^2", axis=ax[2], set_short_x_ticks=True, include_trends=True)
plot_hemisphere_and_global_by_year(shortwave_all_sky-shortwave_clear_sky, shortwave_clear_sky.year, title="CRE Shortwave", 
    fixed_ylim=False, y_label="Flux W/m^2", axis=ax[3], set_short_x_ticks=True, include_trends=True)
plt.savefig("./Figs/"+current_month_folder+"/CERES-Basics")
plt.close()

# Working on Zonal All Sky
zonal_all = create_CERES_20_degree_zonal_data(full_years, "toa_sw_all_mon", start_yr="2001", start_mon="01", 
    end_yr="2022", end_mon="01", remove_leap_year=1)

plot_20_degree_zonal(zonal_all)
plt.title("Global All Sky Zonal Averages")
plt.savefig("./Figs/"+current_month_folder+"/Zonal-All-Sky-Global")
plt.close()

plot_20_degree_zonal(zonal_all, location="sh")
plt.title("SH All Sky Zonal Averages")
plt.savefig("./Figs/"+current_month_folder+"/Zonal-All-Sky-SH")
plt.close()

plot_20_degree_zonal(zonal_all, location="nh")
plt.title("NH All Sky Zonal Averages")
plt.savefig("./Figs/"+current_month_folder+"/Zonal-All-Sky-NH")
plt.close()

# Working on Zonal All Sky
zonal_clr = create_CERES_20_degree_zonal_data(full_years, "toa_sw_clr_c_mon", start_yr="2001", start_mon="01", 
    end_yr="2022", end_mon="01", remove_leap_year=1)

plot_20_degree_zonal(zonal_clr)
plt.title("Global CLear Sky Zonal Averages")
plt.savefig("./Figs/"+current_month_folder+"/Zonal-Clear-Sky-Global")
plt.close()

plot_20_degree_zonal(zonal_clr, location="sh")
plt.title("SH Clear Sky Zonal Averages")
plt.savefig("./Figs/"+current_month_folder+"/Zonal-Clear-Sky-SH")
plt.close()

plot_20_degree_zonal(zonal_clr, location="nh")
plt.title("NH Clear Sky Zonal Averages")
plt.savefig("./Figs/"+current_month_folder+"/Zonal-Clear-Sky-NH")
plt.close()

print("Done")