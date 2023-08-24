from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import scipy
import numpy as np

# Data provided must be sorted by year
def plot_hemisphere_and_global_by_year(data, time_scale, title="NH-SH-Global", 
    y_label=r"Solar Ins. W/m$^2$", fixed_ylim=False, set_x_ticks=True, axis=None,
    set_short_x_ticks=False, include_trends=False):
    if(axis==None):
        axis=plt.gca()
    axis.plot(time_scale, data["global"], 'k-x', label="Global")
    axis.plot(time_scale, data["nh"], 'g-x', label="NH")
    axis.plot(time_scale, data["sh"], 'b-x', label="SH")
    if(set_x_ticks):
        axis.set_xticks([2000, 2003, 2006, 2009, 2012, 2015, 2018, 2021])
    if(set_short_x_ticks):
        axis.set_xticks([2005, 2010, 2015, 2020])
    axis.set_xlabel("Year", fontsize=20)
    if(fixed_ylim):
        axis.set_ylim([339.6, 340.4])
    axis.set_ylabel(y_label, fontsize=20)
    axis.set_title(title, fontsize=20)
    axis.tick_params(axis='both', which='major', labelsize=16)
    axis.legend(fontsize=16, loc="upper right")

    if(include_trends):
        global_results = scipy.stats.linregress(time_scale, data["global"])
        nh_results = scipy.stats.linregress(time_scale, data["nh"])
        sh_results = scipy.stats.linregress(time_scale, data["sh"])
        #text = "Global Trend: "+str(global_coeffs[0].round(3))+" W/m^2 per Year\n", 
        #    "NH Trend: "+str(nh_coeffs[0].round(3))+" W/m^2 per Year\n",
        #    "SH Trend: "+str(sh_coeffs[0].round(3))+" W/m^2 per Year"
        print("Global Trend: "+str(global_results.slope.round(3))+" W/m^2 per Year. With R^2 "+str(global_results.rvalue**2))
        print("NH Trend: "+str(nh_results.slope.round(3))+" W/m^2 per Year")
        print("SH Trend: "+str(sh_results.slope.round(3))+" W/m^2 per Year")
        #axis.text(0.2, 0.2, text)

def plot_data_with_reg_line(time, data, title="Data With Regression", label="", tick_color="b",
    y_label="W/m^2", set_x_ticks=True, axis=None, set_short_x_ticks=False, include_trends=False):
    if(axis==None):
        axis=plt.gca()
    if(label != None):
        axis.plot(time, data, color=tick_color, marker="x")
    else:
        axis.plot(time, data, color=tick_color, marker="x", label=label)

    if(set_x_ticks):
        axis.set_xticks([2000, 2003, 2006, 2009, 2012, 2015, 2018, 2021])
    if(set_short_x_ticks):
        axis.set_xticks([2000, 2005, 2010, 2015, 2020])
    axis.set_xlabel("Year")
    axis.set_ylabel(y_label)
    axis.set_title(title)

    if(include_trends):
        reg_results = scipy.stats.linregress(time, data)
        print(title+" "+str(reg_results.slope.round(3))+" W/m^2 per Year. With R^2 "+str(reg_results.rvalue**2))
        x = np.linspace(time[0], time[-1], 200)
        line = reg_results.slope*x+ reg_results.intercept
        axis.plot(x,line,color=tick_color, linestyle='-', label=label+" Trend={:.3f} with R^2={:.3f} and p-value={:.3f}".format(reg_results.slope, reg_results.rvalue**2, reg_results.pvalue))
        axis.legend()

def plot_spatial_data_trends(data, title=""):
    plt.figure(figsize=(12,6))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.coastlines()
    ax.gridlines()
    data.plot(ax=ax, transform=ccrs.PlateCarree())
    plt.show()

def plot_spatial_all_trend_data(data, title=""):
    fig, ax = plt.subplots(3,1)
    ax[0] = plt.axes(projection=ccrs.Robinson())
    ax[0].coastlines()
    data["trend"].plot(ax=ax[0], transform=ccrs.PlateCarree())

    ax[1] = plt.axes(projection=ccrs.Robinson())
    ax[1].coastlines()
    data["pValue"].plot(ax=ax[1], transform=ccrs.PlateCarree())

    ax[2] = plt.axes(projection=ccrs.Robinson())
    ax[2].coastlines()
    data["rSquared"].plot(ax=ax[2], transform=ccrs.PlateCarree())

def plot_zonal_averages_from_global(global_polyfit_trend_data):
    zonal_dat = global_polyfit_trend_data.mean("lon")
    plt.figure(figsize=(12,6))
    zonal_dat.plot()
