from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import scipy

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