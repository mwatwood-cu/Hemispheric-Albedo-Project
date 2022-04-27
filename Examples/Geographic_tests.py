import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# My built function imports
from data_manipulation_functions import *
from essential_functions import *
from plotting_functions import *

years = "21"

file_name = "./Data/CERES_EBAF-TOA_Full.nc"
#file_name = "./Data/CERES_EBAF-TOA_10years.nc"

ds = xr.open_dataset(file_name)
sw_clr = ds['toa_sw_clr_c_mon']

plot_global_data(sw_clr[0])