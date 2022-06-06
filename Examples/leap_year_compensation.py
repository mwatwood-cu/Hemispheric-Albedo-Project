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
jan_start_full_data = -125*np.cos((x+11)*2*np.pi/year_length)
# Shifted 10 days from equinox to Apr 1
apr_start_full_data = 125*np.sin((x+10)*2*np.pi/year_length)

full_data = jan_start_full_data

monthly_avg = np.zeros(48)
plot_location = np.zeros(48)

for i in range(48):
    # Create logical arrays of indices
    x_range_low = x>=cumulative_days_index[i]
    x_range_high = x< cumulative_days_index[i+1]
    # Use a logical and operation to get both of the above criteria
    x_range = x_range_high & x_range_low

    monthly_avg[i] = np.mean(full_data[x_range])
    plot_location[i] = (cumulative_days_index[i]+cumulative_days_index[i+1])/2

yearly_avg = np.zeros((4,5))
yearly_loc = np.array([1,2,3,4])*365/2
for i in range(4):
    # No yearly weighting
    yearly_avg[i,0] = np.mean(monthly_avg[i*12:(i+1)*12])

    # Weighting with days of the month
    year_total_days = np.sum(days_in_month[i*12:(i+1)*12])
    time_weights = days_in_month[i*12:(i+1)*12]/year_total_days
    yearly_avg[i,1] = np.sum(time_weights*monthly_avg[i*12:(i+1)*12])

    # Correcting Leap year with 365.25 days forced.
    fixed_days_in_month = days_in_month.values.astype("float")
    fixed_days_in_month[fixed_days_in_month==28] = 28.25
    fixed_days_in_month[fixed_days_in_month==29] = 28.25
    time_weights_fixed = fixed_days_in_month[i*12:(i+1)*12]/year_total_days.data
    yearly_avg[i,2] = np.sum(time_weights_fixed*monthly_avg[i*12:(i+1)*12])

    # Correcting Leap year with 365.25 days forced.
    fixed_days_in_month = days_in_month.values.astype("float")
    fixed_days_in_month[fixed_days_in_month==28] = 28.45
    fixed_days_in_month[fixed_days_in_month==29] = 27.65
    time_weights_fixed = fixed_days_in_month[i*12:(i+1)*12]/year_total_days.data
    yearly_avg[i,3] = np.sum(time_weights_fixed*monthly_avg[i*12:(i+1)*12])

    #Averaging by day
    # Create logical arrays of indices
    x_range_low = x>=i*year_length
    x_range_high = x<(i+1)*year_length
    # Use a logical and operation to get both of the above criteria
    x_range = x_range_high & x_range_low
    yearly_avg[i,4] = np.mean(full_data[x_range])

plt.plot(x, full_data)
plt.plot(plot_location, monthly_avg, 'kx')
plt.plot([0,leap_cycle], [0,0],'b')
plt.close()
plt.plot(yearly_loc,yearly_avg[:,0]/125*100,'ro', label="No weighting")
plt.plot(yearly_loc,yearly_avg[:,1]/125*100,'bx', label="Normal weighting")
plt.plot(yearly_loc,yearly_avg[:,2]/125*100,'gx', label="1/4 Day weighting")
plt.plot(yearly_loc,yearly_avg[:,3]/125*100,'cx', label="Proper Day weighting")
plt.plot(yearly_loc,yearly_avg[:,4]/125*100,'k-', label="Proper Day weighting")

print("Done")