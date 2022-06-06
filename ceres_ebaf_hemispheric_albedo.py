#!/usr/bin/env python

"""
====================================================================================================
Investigate hemispheric albedo symmetry using CERES EBAF.

Created by Jake Gristey (jake.j.gristey@noaa.gov) Apr 2021
====================================================================================================
"""

#-------------------------------------------ADD IMPORTS--------------------------------------------#
import numpy as np
import netCDF4
import matplotlib
from matplotlib.ticker import MultipleLocator,AutoMinorLocator,MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, datetime, timedelta
from calendar import monthrange

#-----------------------------------------DEFINE FUNCTIONS-----------------------------------------#
def main():
    """
    Direct execution code.
    """
    #*****************************************USER INPUT*******************************************#
    ebaf_fname='/export/home/jgristey/data/CERES/CERES_EBAF-TOA_Ed4.1_Subset_200003-202011.nc'
    zonal_weights_fname='/export/home/jgristey/data/CERES/CERES_EBAF_zonal_weights.txt'
    #**********************************************************************************************#

    # read data
    nc=netCDF4.Dataset(ebaf_fname)
    lat=nc.variables['lat'][:]
    lon=nc.variables['lon'][:]
    time=nc.variables['time'][:] # days since 2000-03-01 00:00:00
    sw_in=nc.variables['solar_mon'][:]
    sw_out=nc.variables['toa_sw_all_mon'][:]
    nc.close()
    z_weights=np.genfromtxt(zonal_weights_fname,skip_header=17,skip_footer=1)[:,1]

    # convert time dimension to datetime objects
    start=date(2000,3,1) # manually entered from time units in netCDF file
    delta=[timedelta(int(x)) for x in time]
    time=np.array([start+x for x in delta])    

    # zonal, hemispheric, and global means
    sw_in_z=np.mean(sw_in,axis=-1)
    sw_out_z=np.mean(sw_out,axis=-1)
    sw_in_nh=np.array([np.average(sw_in_z[x,90:],weights=z_weights[90:]) for x in \
        range(sw_in_z.shape[0])])
    sw_out_nh=np.array([np.average(sw_out_z[x,90:],weights=z_weights[90:]) for x in \
        range(sw_out_z.shape[0])])
    sw_in_sh=np.array([np.average(sw_in_z[x,:90],weights=z_weights[:90]) for x in \
        range(sw_in_z.shape[0])])
    sw_out_sh=np.array([np.average(sw_out_z[x,:90],weights=z_weights[:90]) for x in \
        range(sw_out_z.shape[0])])
    sw_in_g=np.array([np.average(sw_in_z[x,:],weights=z_weights[:]) for x in \
        range(sw_in_z.shape[0])])
    sw_out_g=np.array([np.average(sw_out_z[x,:],weights=z_weights[:]) for x in \
        range(sw_in_z.shape[0])])

    # 12 month running means (weight by number of days in each month)
    mm=(list(np.arange(1,13))*21)[2:-1] # list of months (hard coded for now)
    yyyy=[2000]*10+list(np.array([[x]*12 for x in range(2001,2020)]).flatten())+[2020]*11 # years
    days=[monthrange(int(yyyy[x]),int(mm[x]))[1] for x in range(len(mm))] # num days per month
    sw_in_g_12mrm=np.array([np.average(sw_in_g[x:x+12],weights=days[x:x+12])for x in \
        range(len(sw_in_g)-12)])
    sw_in_nh_12mrm=np.array([np.average(sw_in_nh[x:x+12],weights=days[x:x+12])for x in \
        range(len(sw_in_nh)-12)])
    sw_in_sh_12mrm=np.array([np.average(sw_in_sh[x:x+12],weights=days[x:x+12])for x in \
        range(len(sw_in_sh)-12)])
    sw_out_g_12mrm=np.array([np.average(sw_out_g[x:x+12],weights=days[x:x+12])for x in \
        range(len(sw_out_g)-12)])
    sw_out_nh_12mrm=np.array([np.average(sw_out_nh[x:x+12],weights=days[x:x+12])for x in \
        range(len(sw_out_nh)-12)])
    sw_out_sh_12mrm=np.array([np.average(sw_out_sh[x:x+12],weights=days[x:x+12])for x in \
        range(len(sw_out_sh)-12)])
    alb_g_12mrm=sw_out_g_12mrm/sw_in_g_12mrm
    alb_nh_12mrm=sw_out_nh_12mrm/sw_in_nh_12mrm
    alb_sh_12mrm=sw_out_sh_12mrm/sw_in_sh_12mrm

    # set plot defaults
    matplotlib.rcParams['axes.linewidth']=1.5
    matplotlib.rcParams['font.size']=14

    # plot incoming solar timeseries
    plt.figure(figsize=(8,4))
    ax=plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    #ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(which='major',direction='out', length=8, width=1.5)
    ax.tick_params(which='minor',direction='out', length=4, width=1.5)
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.set_xlabel('Time [year]')
    ax.set_ylabel('TOA incoming solar irradaince [W m$^{-2}$]')
    plt.plot(time[6:-6],sw_in_nh_12mrm,'g',label='NH mean')
    plt.plot(time[6:-6],sw_in_sh_12mrm,'b',label='SH mean')
    plt.plot(time[6:-6],sw_in_g_12mrm,'k',lw=2,label='Glob mean')
    plt.legend(prop={'size': 10},fancybox=True, framealpha=0.9)
    plt.tight_layout()

    # plot reflected solar timeseries
    plt.figure(figsize=(8,4))
    ax=plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    #ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(which='major',direction='out', length=8, width=1.5)
    ax.tick_params(which='minor',direction='out', length=4, width=1.5)
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel('Time [year]')
    ax.set_ylabel('TOA reflected solar irradaince [W m$^{-2}$]')
    plt.plot(time[6:-6],sw_out_nh_12mrm,'g',label='NH mean')
    plt.plot(time[6:-6],sw_out_sh_12mrm,'b',label='SH mean')
    plt.plot(time[6:-6],sw_out_g_12mrm,'k',lw=2,label='Glob mean')
    plt.legend(prop={'size': 10},fancybox=True, framealpha=0.9)
    plt.tight_layout()

    # plot albedo timeseries
    plt.figure(figsize=(8,4))
    ax=plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    #ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(which='major',direction='out', length=8, width=1.5)
    ax.tick_params(which='minor',direction='out', length=4, width=1.5)
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.yaxis.set_minor_locator(MultipleLocator(0.0002))
    ax.set_xlabel('Time [year]')
    ax.set_ylabel('Albedo')
    plt.plot(time[6:-6],alb_nh_12mrm,'g',label='NH mean')
    plt.plot(time[6:-6],alb_sh_12mrm,'b',label='SH mean')
    plt.plot(time[6:-6],alb_g_12mrm,'k',lw=2,label='Glob mean')
    plt.legend(prop={'size': 10},fancybox=True, framealpha=0.9)
    plt.tight_layout()

    # plot hemispheric albedo difference timeseries
    plt.figure(figsize=(8,4))
    ax=plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    #ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(which='major',direction='out', length=8, width=1.5)
    ax.tick_params(which='minor',direction='out', length=4, width=1.5)
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.yaxis.set_minor_locator(MultipleLocator(0.0002))
    ax.set_xlabel('Time [year]')
    ax.set_ylabel('Albedo difference')
    plt.plot(time[6:-6],alb_nh_12mrm-alb_sh_12mrm,'r',lw=2,label='NH mean minus SH mean')
    plt.legend(prop={'size': 10},fancybox=True, framealpha=0.9)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
