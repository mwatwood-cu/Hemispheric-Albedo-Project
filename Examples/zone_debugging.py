# Installed
import xarray as xr
import sys, os
#from Functions.data_functions import *

# Local Imports
curdir = os.getcwd()
sys.path.insert(0, curdir+"/Functions")
# My built function imports
from data_functions import *
from plotting_functions import *

current_month_folder="June22"
data_path = "/Users/mawa7160/dev/data/CERES/EBAF/"

file_name = data_path+"CERES_EBAF-TOA_Full_2022_01.nc"
ds = xr.open_dataset(file_name)
full_years = ds

zone_data = create_CERES_EBAF_zonal_data(ds, "toa_sw_all_mon", 10)
plot_single_year_by_latitude_from_zonal_average(zone_data, 2001)
plot_zonal_averaged_data_by_year(zone_data)