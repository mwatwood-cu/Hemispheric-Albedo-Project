from cProfile import label
from pandas.core.base import SpecificationError
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy

from data_manipulation_functions import *


def plot_time_series_with_mean_std_once_per_year(time_series, title="", grid=True, axes=None):

    sorted_data_frame = create_yearly_sorted_data(time_series)

    start_year = str(time_series[0].time.dt.year.data)
    end_year = str(time_series[-1].time.dt.year.data)
    dates = make_datetime_once_per_year(start_year, end_year)
    
    means = sorted_data_frame.mean().values
    stds = sorted_data_frame.std().values
    if axes==None :
        f, axes = plt.subplots()

    time_series.plot(ax=axes)
    axes.plot(dates, means)
    axes.fill_between(dates, means-stds, means+stds,color='blue', alpha=0.2)
    axes.set_title(title)
    axes.set_xlabel("")
    if grid:
        axes.grid()

    return

def plot_global_data(spatial_data):
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines()
    spatial_data.plot(ax=ax, transform=ccrs.PlateCarree())
    plt.show()

# Function to show a 3x3 subplot with the Shortwave, Longwave, and Net values
# of radiation for both hemispheres and global average
def plot_CERES_sw_lw_net(sw_all, sw_clear, lw_all, lw_clear, net_all, net_clear):
    #Subplot structure
    fig, ax = plt.subplots(3,3)

    #Shortwave All Sky
    ax[0,0].plot(sw_all.year, sw_all["nh"], '--g', label="NH")
    ax[0,0].plot(sw_all.year, sw_all["sh"], '--b', label="SH")
    ax[0,0].plot(sw_all.year, sw_all["global"], '--k', label="Global")
    ax[0,0].set_title("SW All Sky")
    ax[0,0].set_xticks([])

    #Shortwave Clear Sky
    ax[0,1].plot(sw_clear.year, sw_clear["nh"], '--g')
    ax[0,1].plot(sw_clear.year, sw_clear["sh"], '--b')
    ax[0,1].plot(sw_clear.year, sw_clear["global"], '--k')
    ax[0,1].set_title("SW Clear Sky")
    ax[0,1].set_xticks([])

    #Shortwave Cloud Radiative Effect (CRE)
    ax[0,2].plot(sw_clear.year, sw_all["nh"]-sw_clear["nh"], '--g')
    ax[0,2].plot(sw_clear.year, sw_all["sh"]-sw_clear["sh"], '--b')
    ax[0,2].plot(sw_clear.year, sw_all["global"]-sw_clear["global"], '--k')
    ax[0,2].set_title("SW Cloud Radiative Effect (All-Clear)")
    ax[0,2].set_xticks([])

    #Longwave All Sky
    ax[1,0].plot(lw_all.year, lw_all["nh"], '--g')
    ax[1,0].plot(lw_all.year, lw_all["sh"], '--b')
    ax[1,0].plot(lw_all.year, lw_all["global"], '--k')
    ax[1,0].set_title("LW All Sky")
    ax[1,0].set_xticks([])

    #Longwave Clear Sky
    ax[1,1].plot(lw_clear.year, lw_clear["nh"], '--g')
    ax[1,1].plot(lw_clear.year, lw_clear["sh"], '--b')
    ax[1,1].plot(lw_clear.year, lw_clear["global"], '--k')
    ax[1,1].set_title("LW Clear Sky")
    ax[1,1].set_xticks([])

    #Longwave Cloud Radiative Effect (CRE)
    ax[1,2].plot(sw_clear.year, lw_all["nh"]-lw_clear["nh"], '--g', label="NH")
    ax[1,2].plot(sw_clear.year, lw_all["sh"]-lw_clear["sh"], '--b', label="SH")
    ax[1,2].plot(sw_clear.year, lw_all["global"]-lw_clear["global"], '--k', label="Global")
    ax[1,2].set_title("LW Cloud Radiative Effect")
    ax[1,2].set_xticks([])
    ax[1,2].legend(bbox_to_anchor=(1, 0.75))

    #Net All Sky
    ax[2,0].plot(net_all.year, net_all["nh"], '--g')
    ax[2,0].plot(net_all.year, net_all["sh"], '--b')
    ax[2,0].plot(net_all.year, net_all["global"], '--k')
    ax[2,0].set_title("Net All Sky")

    #Net Clear Sky
    ax[2,1].plot(net_clear.year, net_clear["nh"], '--g')
    ax[2,1].plot(net_clear.year, net_clear["sh"], '--b')
    ax[2,1].plot(net_clear.year, net_clear["global"], '--k')
    ax[2,1].set_title("Net Clear Sky")

    #Net Cloud Radiative Effect (CRE)
    ax[2,2].plot(sw_clear.year, net_all["nh"]-net_clear["nh"], '--g')
    ax[2,2].plot(sw_clear.year, net_all["sh"]-net_clear["sh"], '--b')
    ax[2,2].plot(sw_clear.year, net_all["global"]-net_clear["global"], '--k')
    ax[2,2].set_title("Net Cloud Radiative Effect")

# Function to show a 2x4 subplot with the Shortwave, Longwave, and Net values
# level of asymmetry. Subtracting NH-SH.
def plot_CERES_asymmetry(sw_all, sw_clear, lw_all, lw_clear, net_all, net_clear, solar, 
    cloud_area, cloud_pressure, cloud_temp, cloud_tau):
       #Subplot structure
    fig, ax = plt.subplots(2,4)

    # Solar Input
    ax[0,0].plot(sw_all.year, (solar["nh"]-solar["sh"])/solar["global"].mean()*100, '--y', label="Solar In")
    ax[0,0].set_title("Solar Insolation")
    ax[0,0].set_xticks([])
    ax[0,0].set_ylabel("Asymmetry Relative to Global Mean Solar")

    #All Sky Asymmetry
    ax[0,1].plot(sw_all.year, (sw_all["nh"]-sw_all["sh"])/solar["global"].mean()*100, '--g', label="SW")
    ax[0,1].plot(lw_all.year, (lw_all["nh"]-lw_all["sh"])/solar["global"].mean()*100, '--r', label="LW")
    ax[0,1].plot(net_all.year, (net_all["nh"]-net_all["sh"])/solar["global"].mean()*100, '--k', label="Net")
    ax[0,1].set_title("All Sky")
    ax[0,1].set_xticks([])

    # Clear Sky Asymmetry
    ax[0,2].plot(sw_clear.year, (sw_clear["nh"]-sw_clear["sh"])/solar["global"].mean()*100, '--g', label="SW")
    ax[0,2].plot(lw_clear.year, (lw_clear["nh"]-lw_clear["sh"])/solar["global"].mean()*100, '--r', label="LW")
    ax[0,2].plot(net_clear.year, (net_clear["nh"]-net_clear["sh"])/solar["global"].mean()*100, '--k', label="Net")
    ax[0,2].set_title("Clear Sky")
    ax[0,2].set_xticks([])

    # CRE Asymmetry
    ax[0,3].plot(sw_all.year, ((sw_all["nh"]-sw_clear["nh"])-(sw_all["sh"]-sw_clear["sh"]))/solar["global"].mean()*100, '--g', label="SW")
    ax[0,3].plot(lw_all.year, ((lw_all["nh"]-lw_clear["nh"])-(lw_all["sh"]-lw_clear["sh"]))/solar["global"].mean()*100, '--r', label="LW")
    ax[0,3].plot(net_all.year, ((net_all["nh"]-net_clear["nh"])-(net_all["sh"]-net_clear["sh"]))/solar["global"].mean()*100, '--k', label="Net")
    ax[0,3].set_title("Cloud Radiative Effect")
    ax[0,3].legend(bbox_to_anchor=(1, 0.75))
    ax[0,3].set_xticks([])

    # CERES Cloud Properties
    ax[1,0].plot(sw_all.year, (cloud_area["nh"]-cloud_area["sh"])/cloud_area["global"].mean()*100, '--b', label="Cloud Area")
    ax[1,0].set_title("Cloud Area")
    ax[1,0].set_ylabel("Asymmetry Relative to Global Mean")
    #ax[1,0].set_xticks([])

    #CERES Cloud Temp
    ax[1,1].plot(sw_all.year, (cloud_temp["nh"]-cloud_temp["sh"])/cloud_temp["global"].mean()*100, '--m', label="Cloud Temp")
    ax[1,1].set_title("Cloud Temperature")
    #ax[1,0].set_xticks([])

    #CERES Cloud Temp
    ax[1,2].plot(sw_all.year, (cloud_pressure["nh"]-cloud_pressure["sh"])/cloud_pressure["global"].mean()*100, '--m', label="Cloud Pressure")
    ax[1,2].set_title("Cloud Pressure")
    #ax[1,0].set_xticks([])

    # Cloud Optical Depth
    ax[1,3].plot(sw_clear.year, (cloud_tau["nh"]-cloud_tau["sh"])/cloud_tau["global"].mean()*100, '--k', label="Cloud Optical Depth")
    ax[1,3].set_title("Cloud Optical Depth")
    #ax[1,1].set_xticks([])

# Data provided must be sorted by year
def plot_hemisphere_and_global_by_year(data, title="NH-SH-Global", y_label="Solar Insolation W/m^2", fixed_ylim=False):
    plt.plot(data.year, data["global"], 'k-x', label="Global")
    plt.plot(data.year, data["nh"], 'g-x', label="NH")
    plt.plot(data.year, data["sh"], 'b-x', label="SH")
    plt.xticks([2000, 2003, 2006, 2009, 2012, 2015, 2018, 2021])
    plt.xlabel("Year")
    if(fixed_ylim):
        plt.ylim([339.6, 340.4])
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()