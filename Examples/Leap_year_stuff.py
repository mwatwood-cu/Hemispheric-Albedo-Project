import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sys, os
from sklearn.metrics import mean_squared_error

LEAP_YEAR_CORRECTION = (366/1461)/0.25
NON_LEAP_YEAR_CORRECTION = (365/1461)/0.25

curdir = os.getcwd()
sys.path.insert(0, curdir+"/Functions")

# My built function imports
from data_functions import *
from plotting_functions import *

file_name = "./Data/CERES_EBAF-TOA_Full.nc"
ds = xr.open_dataset(file_name)

full_years = clean_CERES_data(ds)

solar_in = create_CERES_hemisphere_data(full_years, "solar_mon")

solar_in_no_fix = create_CERES_hemisphere_data(full_years, "solar_mon", remove_leap_year=False)

### Another idea for leap year corrections
'''
manual_corrected_nh = solar_in_no_fix["nh"].copy()
manual_corrected_sh = solar_in_no_fix["sh"].copy()
four_year_avg = np.zeros((3,5))
year_index = 0
running_avg = np.zeros(3)

for index, current_year in enumerate(manual_corrected_nh.year.values):
    running_avg[0] += manual_corrected_nh[index].values
    running_avg[1] += manual_corrected_sh[index].values
    running_avg[2] += solar_in_no_fix["global"][index].values
    if(current_year%4==0):
        manual_corrected_nh[index] = manual_corrected_nh[index]*LEAP_YEAR_CORRECTION
        manual_corrected_sh[index] = manual_corrected_sh[index]*LEAP_YEAR_CORRECTION

        four_year_avg[0,year_index] = running_avg[0]/4
        four_year_avg[1,year_index] = running_avg[1]/4
        four_year_avg[2,year_index] = running_avg[2]/4

        running_avg = np.zeros(3)
        year_index += 1
    else:
        manual_corrected_nh[index] = manual_corrected_nh[index]*NON_LEAP_YEAR_CORRECTION
        manual_corrected_sh[index] = manual_corrected_sh[index]*NON_LEAP_YEAR_CORRECTION
'''

### Find minimum values for February Corrections
'''
day_range = np.arange(28.4, 28.6, 0.01)
minimum = 100000
min_leap = 27
min_non_leap = 27
print("Starting")
for i in range(len(day_range)):
    nl_length = day_range[i]
    l_length = 113-nl_length*3
    solar_test = create_CERES_hemisphere_data(full_years, "solar_mon", leap_year_correction=l_length, 
        non_leap_year_correction=nl_length)
    current_error = mean_squared_error(solar_test["global"], solar_test["nh"])
    if(current_error<minimum):
        minimum = current_error
        min_leap = l_length
        min_non_leap = nl_length
    #print("Done with "+str(day_range[i]))

print("Non Leap Year")
print(min_non_leap)
print("Leap Year")
print(min_leap)
print(minimum)
'''

a=1