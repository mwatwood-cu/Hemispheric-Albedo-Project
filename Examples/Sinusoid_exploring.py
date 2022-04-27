import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Data Reference
file_name = "./Data/CERES_EBAF-TOA_Full.nc"
ds = xr.open_dataset(file_name)
days_in_month = ds.time.dt.days_in_month
# Take days in the month starting in April
days_in_month = days_in_month[0:48]
cumulative_days_index = np.cumsum(days_in_month)-1
cumulative_days_index = np.concatenate(([0], cumulative_days_index))

# Length of time in years
year_length = 365.242189
leap_cycle = year_length*4
x = np.linspace(0, leap_cycle, 100000)

# Shifted 11 days from solstice to Jan 1
jan_start_full_data = -125*np.cos((x+11)*2*np.pi/year_length)+200
# Shifted 10 days from equinox to Apr 1
apr_start_full_data = 125*np.sin((x+10)*2*np.pi/year_length)+200

# End of year Average test for December
dec_1_yr_1 = x>= 334
dec_31_yr_1 = x<= 365
full_dec = x<= 365.242189
x_range = dec_1_yr_1 & dec_31_yr_1
x_full_range = dec_1_yr_1 & full_dec

# Calculate Average
dec_avg = jan_start_full_data[x_range].mean()
dec_corrected_avg = jan_start_full_data[x_full_range].mean()
print("January Start Average")
print(31/(dec_avg/dec_corrected_avg))

# Calculate area
dec_area = np.trapz(jan_start_full_data[x_range], x[x_range])
dec_cor_area = np.trapz(jan_start_full_data[x_full_range], x[x_full_range])
print("January Start Area")
print(31/(dec_area/dec_cor_area))

# End of year Average test for December
feb_1_yr_1 = x>= 337
feb_31_yr_1 = x<= 365
full_feb = x<= year_length
x_range = feb_1_yr_1 & feb_31_yr_1
x_full_range = feb_1_yr_1 & full_feb

# Calculate Average
feb_avg = apr_start_full_data[x_range].mean()
feb_corrected_avg = apr_start_full_data[x_full_range].mean()
print("April Start Average")
print(28/(feb_avg/feb_corrected_avg))

# Calculate area
feb_area = np.trapz(apr_start_full_data[x_range], x[x_range])
feb_cor_area = np.trapz(apr_start_full_data[x_full_range], x[x_full_range])
print("April January Start Area")
print(28/(feb_area/feb_cor_area))

print("done")