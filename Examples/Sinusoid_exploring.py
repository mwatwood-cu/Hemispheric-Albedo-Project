import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def calculate_monthly_avg_four_years(x, data, cum_days):
    monthly_data = np.zeros(48)
    monthly_locs = np.zeros(48)

    for i in range(48):
        start_x = x >= cum_days[i]
        end_x = x<= cum_days[i+1]
        x_range = end_x & start_x
        this_month = data[x_range]

        monthly_data[i] = np.mean(this_month)
        monthly_locs[i] = (cum_days[i+1]+cum_days[i])/2
    
    return monthly_data, monthly_locs


def create_full_data(data_set, start_mon):
    days_in_month = data_set.time.dt.days_in_month

    # Take days in the month starting in April
    days_in_month = days_in_month[start_mon+9:start_mon+57]
    cumulative_days_index = np.cumsum(days_in_month)-1
    cumulative_days_index = np.concatenate(([0], cumulative_days_index))

    # Length of time in years
    year_length = 365.242189
    leap_cycle = year_length*4
    x = np.linspace(0, leap_cycle, 100000)

    amplitude = 125
    vertical_offset = 200
    solstice_offset = 11

    total_offset = solstice_offset+cumulative_days_index[start_mon]
    print(total_offset)

    # Shifted 11 days from solstice to Jan 1
    nh_full_data = -amplitude*np.cos((x+total_offset)*2*np.pi/year_length)+vertical_offset
    sh_full_data = amplitude*np.cos((x+total_offset)*2*np.pi/year_length)+vertical_offset

    nh_months, nh_times = calculate_monthly_avg_four_years(x, nh_full_data, cumulative_days_index)
    sh_months, sh_times = calculate_monthly_avg_four_years(x, sh_full_data, cumulative_days_index)

    plt.plot(x, nh_full_data,'b-')
    plt.plot(nh_times, nh_months,'bo')

    year_dat = np.zeros((4,2))
    for i in range(4):
        year_dat[i, 0] = np.average(nh_months[i*12:(i+1)*12], weights=days_in_month[i*12:(i+1)*12])
        year_dat[i, 1] = np.average(sh_months[i*12:(i+1)*12], weights=days_in_month[i*12:(i+1)*12])

    return year_dat

# Data Reference
file_name = "./Data/CERES_EBAF-TOA_Full_2022_01.nc"
ds = xr.open_dataset(file_name)

years = create_full_data(ds, 0)

print("done")

'''
start_mon = 0
total_offset = solstice_offset+cumulative_days_index[start_mon+1]
jan_start_full_data = -amplitude*np.cos((x+total_offset)*2*np.pi/year_length)+vertical_offset

start_mon = 3
total_offset = solstice_offset+cumulative_days_index[start_mon+1]
apr_start_full_data = -amplitude*np.cos((x+total_offset)*2*np.pi/year_length)+vertical_offset

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
print("April Start Area")
print(28/(feb_area/feb_cor_area))

plt.plot(x, jan_start_full_data)
plt.plot()
'''

print("done")

